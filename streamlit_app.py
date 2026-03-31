"""
Insurance Review Analyzer — Streamlit Application
Covers: Prediction, Summary, Explanation, Information Retrieval, RAG, QA
"""

import streamlit as st
import pandas as pd
import numpy as np
import json, os

st.set_page_config(page_title="Insurance Review Analyzer", layout="wide")

# ═══════════════════════════════════════════════════
# DATA & MODEL LOADING (cached)
# ═══════════════════════════════════════════════════

@st.cache_data
def load_data():
    df = pd.read_csv("data/topics.csv")
    embeddings = np.load("data/sentence_embeddings.npy")
    with open("data/05_metrics.json") as f:
        metrics = json.load(f)
    return df, embeddings, metrics


@st.cache_resource
def load_prediction_models():
    import joblib
    tfidf = joblib.load("data/05_tfidf_vectorizer.joblib")
    logreg = joblib.load("data/05_tfidf_logreg.joblib")
    return tfidf, logreg


@st.cache_resource
def load_topic_classifier():
    """Train a lightweight topic classifier on the fly."""
    from sklearn.linear_model import LogisticRegression
    df, _, _ = load_data()
    tfidf, _ = load_prediction_models()
    labeled = df[df["lda_topic_label"].notna()]
    X = tfidf.transform(labeled["avis_corrected"].fillna(""))
    y = labeled["lda_topic_label"]
    model = LogisticRegression(max_iter=500, C=1.0, solver="lbfgs", n_jobs=-1)
    model.fit(X, y)
    return model


@st.cache_resource
def load_sentence_model():
    from sentence_transformers import SentenceTransformer
    return SentenceTransformer("paraphrase-multilingual-MiniLM-L12-v2")


@st.cache_resource
def load_generator():
    from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained("google/flan-t5-small")
    model = AutoModelForSeq2SeqLM.from_pretrained("google/flan-t5-small")
    return model, tokenizer


def generate_text(prompt, max_new_tokens=150):
    model, tokenizer = load_generator()
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512)
    outputs = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)
    return tokenizer.decode(outputs[0], skip_special_tokens=True)


def semantic_search(query_text, embeddings, model_st, top_k=5):
    """Return indices and cosine similarities of top-k matches."""
    q_emb = model_st.encode([query_text])[0]
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    norms = np.where(norms == 0, 1, norms)
    sims = (embeddings / norms) @ (q_emb / (np.linalg.norm(q_emb) + 1e-10))
    top_idx = np.argsort(sims)[::-1][:top_k]
    return top_idx, sims[top_idx]


# ═══════════════════════════════════════════════════
# MAIN APP
# ═══════════════════════════════════════════════════

st.title("Insurance Review Analyzer")
st.caption("NLP-powered analysis of 34,428 French insurance reviews")

df, embeddings, metrics = load_data()

tabs = st.tabs(["Prediction", "Summary", "Explanation", "Search",
                "RAG", "QA"])

# ─────────────────────────────────────────────────
# TAB 1: PREDICTION
# ─────────────────────────────────────────────────
with tabs[0]:
    st.header("Star Rating & Topic Prediction")
    st.write("Enter a French insurance review to predict its star rating and topic.")

    review_text = st.text_area("Review text", height=120,
                               placeholder="Tapez un avis ici...",
                               key="pred_input")

    if st.button("Predict", key="pred_btn") and review_text.strip():
        tfidf, logreg = load_prediction_models()
        topic_model = load_topic_classifier()

        X = tfidf.transform([review_text])
        star_pred = logreg.predict(X)[0]
        star_proba = logreg.predict_proba(X)[0]
        topic_pred = topic_model.predict(X)[0]
        topic_proba = topic_model.predict_proba(X)[0]

        col1, col2 = st.columns(2)

        with col1:
            st.subheader(f"Predicted: {star_pred} / 5")
            proba_df = pd.DataFrame({
                "Stars": ["1", "2", "3", "4", "5"],
                "Probability": star_proba
            })
            st.bar_chart(proba_df.set_index("Stars"))

        with col2:
            st.subheader(f"Topic: {topic_pred}")
            topic_classes = topic_model.classes_
            top3_idx = np.argsort(topic_proba)[::-1][:3]
            for idx in top3_idx:
                st.write(f"- {topic_classes[idx]}: {topic_proba[idx]:.1%}")

    elif review_text.strip() == "":
        pass

# ─────────────────────────────────────────────────
# TAB 2: SUMMARY
# ─────────────────────────────────────────────────
with tabs[1]:
    st.header("Insurer Summary")

    insurers = sorted(df["assureur"].dropna().unique())
    selected = st.selectbox("Select an insurer", insurers, key="sum_insurer")

    if selected:
        sub = df[df["assureur"] == selected]
        labeled = sub[sub["note"].notna()]

        col1, col2, col3 = st.columns(3)
        col1.metric("Total Reviews", len(sub))
        col2.metric("Avg Rating", f"{labeled['note'].mean():.2f}" if len(labeled) > 0 else "N/A")
        col3.metric("Labeled Reviews", len(labeled))

        if len(labeled) > 0:
            # Rating distribution
            st.subheader("Rating Distribution")
            rating_counts = labeled["note"].value_counts().sort_index()
            st.bar_chart(rating_counts)

            # Topic breakdown
            st.subheader("Topic Breakdown")
            topic_stats = labeled.groupby("lda_topic_label").agg(
                count=("note", "size"),
                avg_rating=("note", "mean")
            ).sort_values("count", ascending=False)
            topic_stats["avg_rating"] = topic_stats["avg_rating"].round(2)
            st.dataframe(topic_stats, use_container_width=True)

            # Representative reviews (extractive summary)
            st.subheader("Representative Reviews")
            # Pick 5 reviews closest to the insurer's embedding centroid
            sub_idx = sub.index.tolist()
            valid_idx = [i for i in sub_idx if i < len(embeddings)]
            if valid_idx:
                sub_embs = embeddings[valid_idx]
                centroid = sub_embs.mean(axis=0)
                dists = np.linalg.norm(sub_embs - centroid, axis=1)
                closest = np.argsort(dists)[:5]
                for rank, ci in enumerate(closest, 1):
                    row = df.iloc[valid_idx[ci]]
                    stars = f"{row['note']:.0f}/5" if pd.notna(row["note"]) else "N/A"
                    st.markdown(f"**{rank}.** ({stars}) {row['avis_corrected'][:300]}...")

            # Abstractive summary with flan-t5
            if st.button("Generate AI Summary", key="sum_gen"):
                with st.spinner("Generating summary..."):
                    sample_reviews = labeled.head(10)["avis_en"].fillna("").tolist()
                    context = "\n".join(f"- {r[:200]}" for r in sample_reviews if r.strip())
                    prompt = (f"Summarize the following customer reviews about {selected} "
                              f"insurance company:\n{context}\n\nSummary:")
                    result = generate_text(prompt, max_new_tokens=150)
                    st.info(result)

# ─────────────────────────────────────────────────
# TAB 3: EXPLANATION
# ─────────────────────────────────────────────────
with tabs[2]:
    st.header("Prediction Explanation")
    st.write("See which words drive the star rating prediction (LIME).")

    expl_text = st.text_area("Review text", height=120,
                             placeholder="Tapez un avis ici...",
                             key="expl_input")

    if st.button("Explain", key="expl_btn") and expl_text.strip():
        with st.spinner("Computing LIME explanation (10-15 sec)..."):
            from lime.lime_text import LimeTextExplainer

            tfidf, logreg = load_prediction_models()

            def predict_fn(texts):
                return logreg.predict_proba(tfidf.transform(texts))

            pred = logreg.predict(tfidf.transform([expl_text]))[0]
            proba = logreg.predict_proba(tfidf.transform([expl_text]))[0]

            explainer = LimeTextExplainer(
                class_names=["1", "2", "3", "4", "5"],
                split_expression=r"\W+",
            )
            exp = explainer.explain_instance(
                expl_text, predict_fn,
                num_features=15, num_samples=300,
                labels=[pred - 1],  # 0-indexed
            )

            st.subheader(f"Predicted: {pred} / 5")
            st.write(f"Confidence: {proba[pred-1]:.1%}")

            # Display word contributions
            weights = exp.as_list(label=pred - 1)
            pos_words = [(w, s) for w, s in weights if s > 0]
            neg_words = [(w, s) for w, s in weights if s < 0]

            col1, col2 = st.columns(2)
            with col1:
                st.markdown("**Supports this rating (positive)**")
                for w, s in sorted(pos_words, key=lambda x: -x[1]):
                    st.write(f"  +{s:.3f}  **{w}**")
            with col2:
                st.markdown("**Against this rating (negative)**")
                for w, s in sorted(neg_words, key=lambda x: x[1]):
                    st.write(f"  {s:.3f}  **{w}**")

            # Render LIME HTML with dark-mode fix
            html = exp.as_html()
            dark_css = """<style>
            body, .lime { background-color: transparent !important; color: #fafafa !important; }
            td, th, span, p, div, h1, h2, h3, h4, h5 { color: #fafafa !important; }
            table { border-color: #555 !important; }
            svg text { fill: #fafafa !important; }
            </style>"""
            st.components.v1.html(dark_css + html, height=400, scrolling=True)

# ─────────────────────────────────────────────────
# TAB 4: INFORMATION RETRIEVAL (Search)
# ─────────────────────────────────────────────────
with tabs[3]:
    st.header("Semantic Search")
    st.write("Find similar reviews using sentence embeddings (MiniLM).")

    query = st.text_input("Search query", placeholder="probleme de remboursement",
                          key="search_q")

    col_f1, col_f2, col_f3 = st.columns(3)
    with col_f1:
        filter_stars = st.multiselect("Filter by stars", [1, 2, 3, 4, 5],
                                      default=[], key="search_stars")
    with col_f2:
        filter_insurer = st.selectbox("Filter by insurer", ["All"] + insurers,
                                      key="search_insurer")
    with col_f3:
        top_k = st.slider("Number of results", 3, 20, 5, key="search_k")

    if st.button("Search", key="search_btn") and query.strip():
        with st.spinner("Searching..."):
            model_st = load_sentence_model()

            # Apply filters to get candidate indices
            mask = np.ones(len(df), dtype=bool)
            if filter_stars:
                mask &= df["note"].isin(filter_stars)
            if filter_insurer != "All":
                mask &= df["assureur"] == filter_insurer
            candidate_idx = np.where(mask)[0]

            if len(candidate_idx) == 0:
                st.warning("No reviews match the filters.")
            else:
                cand_embs = embeddings[candidate_idx]
                q_emb = model_st.encode([query])[0]
                norms = np.linalg.norm(cand_embs, axis=1, keepdims=True)
                norms = np.where(norms == 0, 1, norms)
                sims = (cand_embs / norms) @ (q_emb / (np.linalg.norm(q_emb) + 1e-10))
                top_local = np.argsort(sims)[::-1][:top_k]

                for rank, li in enumerate(top_local, 1):
                    gi = candidate_idx[li]
                    row = df.iloc[gi]
                    stars = f"{row['note']:.0f}" if pd.notna(row["note"]) else "?"
                    sim = sims[li]
                    with st.container():
                        st.markdown(
                            f"**#{rank}** — {stars}/5 | {row['assureur']} | "
                            f"{row.get('lda_topic_label', 'N/A')} | sim={sim:.3f}"
                        )
                        st.write(str(row["avis_corrected"])[:500])
                        st.divider()

# ─────────────────────────────────────────────────
# TAB 5: RAG (Retrieval-Augmented Generation)
# ─────────────────────────────────────────────────
with tabs[4]:
    st.header("RAG — Ask a Question About Reviews")
    st.write("Ask an open-ended question. The system retrieves relevant reviews and "
             "generates a synthesis using a language model.")

    rag_query = st.text_input("Your question",
                              placeholder="What do customers think about claims handling?",
                              key="rag_q")
    rag_k = st.slider("Number of reviews to retrieve", 3, 10, 5, key="rag_k")

    if st.button("Generate Answer", key="rag_btn") and rag_query.strip():
        with st.spinner("Retrieving reviews and generating answer..."):
            model_st = load_sentence_model()

            top_idx, sims = semantic_search(rag_query, embeddings, model_st, top_k=rag_k)

            # Build context from English translations
            context_parts = []
            for i, idx in enumerate(top_idx):
                row = df.iloc[idx]
                en_text = str(row.get("avis_en", row["avis_corrected"]))[:200]
                stars = f"{row['note']:.0f}" if pd.notna(row["note"]) else "?"
                context_parts.append(f"[{stars}/5] {en_text}")

            context = "\n".join(context_parts)
            prompt = (f"Based on these customer reviews:\n{context}\n\n"
                      f"Question: {rag_query}\n"
                      f"Provide a detailed answer based on the reviews:")

            result = generate_text(prompt, max_new_tokens=200)

            st.subheader("Answer")
            st.info(result)

            st.subheader("Source Reviews")
            for i, idx in enumerate(top_idx):
                row = df.iloc[idx]
                stars = f"{row['note']:.0f}" if pd.notna(row["note"]) else "?"
                st.markdown(
                    f"**{i+1}.** ({stars}/5, sim={sims[i]:.3f}) — "
                    f"{row['assureur']} — {row.get('lda_topic_label', 'N/A')}"
                )
                st.caption(str(row["avis_corrected"])[:300])

# ─────────────────────────────────────────────────
# TAB 6: QA (Question Answering)
# ─────────────────────────────────────────────────
with tabs[5]:
    st.header("QA — Question Answering")
    st.write("Ask a specific question about an insurer or topic. "
             "The system finds relevant reviews and generates a concise answer.")

    col_qa1, col_qa2 = st.columns([3, 1])
    with col_qa1:
        qa_question = st.text_input("Question",
                                    placeholder="Is Direct Assurance recommended for auto insurance?",
                                    key="qa_q")
    with col_qa2:
        qa_insurer = st.selectbox("Insurer (optional)", ["Any"] + insurers,
                                  key="qa_insurer")

    if st.button("Get Answer", key="qa_btn") and qa_question.strip():
        with st.spinner("Finding answer..."):
            model_st = load_sentence_model()

            # Filter by insurer if specified
            if qa_insurer != "Any":
                mask = df["assureur"] == qa_insurer
                candidate_idx = np.where(mask)[0]
                cand_embs = embeddings[candidate_idx]
                q_emb = model_st.encode([qa_question])[0]
                norms = np.linalg.norm(cand_embs, axis=1, keepdims=True)
                norms = np.where(norms == 0, 1, norms)
                sims = (cand_embs / norms) @ (q_emb / (np.linalg.norm(q_emb) + 1e-10))
                top_local = np.argsort(sims)[::-1][:5]
                top_idx = candidate_idx[top_local]
                top_sims = sims[top_local]
            else:
                top_idx, top_sims = semantic_search(qa_question, embeddings, model_st, top_k=5)

            # Build context
            context_parts = []
            for idx in top_idx:
                row = df.iloc[idx]
                en_text = str(row.get("avis_en", row["avis_corrected"]))[:200]
                stars = f"{row['note']:.0f}" if pd.notna(row["note"]) else "?"
                context_parts.append(f"[{stars}/5] {en_text}")

            context = "\n".join(context_parts)
            prompt = (f"Based on these customer reviews:\n{context}\n\n"
                      f"Answer concisely: {qa_question}\nAnswer:")

            result = generate_text(prompt, max_new_tokens=100)

            st.subheader("Answer")
            st.success(result)

            # Show evidence
            with st.expander("Evidence (retrieved reviews)"):
                for i, idx in enumerate(top_idx):
                    row = df.iloc[idx]
                    stars = f"{row['note']:.0f}" if pd.notna(row["note"]) else "?"
                    st.markdown(f"**{i+1}.** ({stars}/5) {row['assureur']}")
                    st.caption(str(row["avis_corrected"])[:300])
