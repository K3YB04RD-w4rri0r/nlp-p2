# Streamlit Application — Insurance Review Analyzer

## Overview

This application provides an interactive interface for analyzing 34,428 French insurance reviews using NLP models trained in the preceding notebooks (01–05). It is organized into six tabs, each addressing a distinct analytical task.

## How to Run

```bash
pip install streamlit lime sentence-transformers transformers
streamlit run streamlit_app.py
```

Open `http://localhost:8501` in a browser. The first load of each tab may take 10–15 seconds while models are downloaded and cached.

## Required Files

All files are in the `data/` directory:

| File | Source | Description |
|---|---|---|
| `topics.csv` | Notebook 02–03 | Full dataset: 34,428 reviews with text, ratings, topic labels |
| `sentence_embeddings.npy` | Notebook 04 | Pre-computed MiniLM sentence embeddings (34,428 x 384) |
| `05_tfidf_vectorizer.joblib` | Notebook 05 | Fitted TF-IDF vectorizer (30K features, unigrams + bigrams) |
| `05_tfidf_logreg.joblib` | Notebook 05 | Trained Logistic Regression model for star prediction |
| `05_metrics.json` | Notebook 05 | Performance metrics for all trained models |

## Tabs

### 1. Prediction

**Purpose:** Given a French review text, predict its star rating (1–5) and its topic category.

**How it works:**
- The review text is transformed into a TF-IDF vector using the pre-fitted vectorizer from notebook 05 (30,000 features, unigrams and bigrams, sublinear TF scaling).
- A **Logistic Regression** classifier predicts the star rating. This model was trained on 19,279 reviews and achieves a Macro-F1 of 0.465 on the validation set.
- A second Logistic Regression, trained on the fly using LDA topic labels from notebook 03, predicts the review's topic (e.g., "Claims & Damage Handling", "Pricing & Rate Changes").
- Both predictions are displayed with their full probability distributions, so the user can see how confident the model is and which alternative classes were considered.

**Why Logistic Regression:** It was chosen over CamemBERT for the app because it runs instantly on CPU with no GPU requirement, while achieving competitive performance (only 2.6 Macro-F1 points below CamemBERT). The probability calibration of Logistic Regression is also better suited for displaying confidence scores.

---

### 2. Summary

**Purpose:** Provide an analytical overview of a selected insurance company based on its reviews.

**How it works:**
- The user selects an insurer from a dropdown (all insurers in the dataset are available).
- **Key metrics** are displayed: total review count, average star rating, and number of labeled reviews.
- **Rating distribution** is shown as a bar chart (1–5 stars).
- **Topic breakdown** shows a table of LDA topics for that insurer with review counts and average ratings per topic. This reveals what customers talk about most and which topics are associated with higher or lower satisfaction.
- **Representative reviews** are selected using an extractive approach: the MiniLM sentence embeddings (384-dimensional, from notebook 04) are used to compute the centroid of all reviews for that insurer, then the 5 reviews closest to the centroid (by Euclidean distance) are displayed. These are the most "typical" reviews for that insurer.
- **AI Summary** (optional, triggered by button): the 10 most recent labeled reviews are passed (via their English translations) to **Google's Flan-T5-Small** (77M parameters), which generates a short abstractive summary. English translations are used because Flan-T5 performs best in English.

---

### 3. Explanation

**Purpose:** Show which words in a review drive the star rating prediction, making the model's decision transparent and interpretable.

**How it works:**
- The user enters a review and clicks "Explain".
- **LIME (Local Interpretable Model-agnostic Explanations)** is used to explain the prediction. LIME works by:
  1. Generating ~300 perturbations of the input text (randomly removing words)
  2. Getting the Logistic Regression's prediction for each perturbation
  3. Fitting a local linear model to understand which words, when removed, most change the prediction
- The result is a list of words with positive or negative contribution scores:
  - **Positive contributions** push the prediction toward the predicted class
  - **Negative contributions** push the prediction away from it
- Both a structured word list and an interactive LIME HTML visualization are displayed.

**Why LIME over SHAP:** LIME is simpler to apply to text classification and produces intuitive word-level explanations. SHAP requires a background dataset and is significantly slower for text models.

---

### 4. Search (Information Retrieval)

**Purpose:** Find reviews semantically similar to a user query, with optional filtering.

**How it works:**
- The user enters a natural language query in French (e.g., "problème de remboursement après un sinistre").
- The query is encoded into a 384-dimensional vector using **MiniLM** (`paraphrase-multilingual-MiniLM-L12-v2`), the same model used in notebook 04.
- **Cosine similarity** is computed between the query vector and all 34,428 pre-computed review embeddings. This is a matrix-vector multiplication that runs in ~50ms.
- Optional **filters** can restrict results by star rating and/or insurer.
- The top-k results are displayed with their similarity score, star rating, insurer name, and LDA topic.

**Why semantic search over keyword search:** Traditional keyword search (TF-IDF, BM25) requires exact word matches. Semantic search understands meaning — a query about "remboursement" will also find reviews mentioning "indemnisation" or "prise en charge" because MiniLM maps synonyms to nearby vectors.

---

### 5. RAG (Retrieval-Augmented Generation)

**Purpose:** Answer open-ended questions about the review corpus by combining retrieval with text generation.

**How it works:**
- The user enters a question (e.g., "What do customers think about claims handling?").
- **Step 1 — Retrieval:** The question is encoded with MiniLM and the top-k most similar reviews are retrieved via cosine similarity (same mechanism as the Search tab).
- **Step 2 — Generation:** The English translations of the retrieved reviews are concatenated into a context prompt, which is fed to **Flan-T5-Small** with the user's question. The model generates a synthesis paragraph grounded in the actual review content.
- Both the generated answer and the source reviews are displayed, so the user can verify the answer against the evidence.

**Why RAG over pure generation:** A standalone language model would hallucinate facts about insurers it has never seen. RAG grounds the generation in actual customer reviews, producing answers that are factually supported by the data. The source reviews are shown for transparency.

**Why English translations:** Flan-T5-Small is primarily trained on English text. Using the `avis_en` column (English translations from notebook 02) produces significantly better generation quality than feeding it French text directly.

---

### 6. QA (Question Answering)

**Purpose:** Answer specific, focused questions about an insurer or topic, with optional insurer filtering.

**How it works:**
- The user enters a specific question (e.g., "Is Direct Assurance recommended for auto insurance?") and optionally selects an insurer to restrict the search.
- **Retrieval:** If an insurer is selected, only that insurer's reviews are searched. Otherwise, all 34,428 reviews are candidates. The top 5 most relevant reviews are retrieved using the same MiniLM semantic search.
- **Generation:** The retrieved reviews (English translations) are fed to Flan-T5-Small with a concise-answer prompt. The model generates a short, focused response.
- The answer is displayed prominently, with the supporting reviews available in an expandable section ("Evidence").

**Difference from RAG:** The QA tab is designed for specific, focused questions (yes/no, factual) with an optional insurer filter, while RAG is designed for broader, open-ended synthesis across the full corpus. The generation prompt is also different — QA asks for a concise answer, RAG asks for a detailed synthesis.

---

## Architecture

```
User Input
    |
    v
[Streamlit UI] ---> Tab Router
    |
    +--- Prediction ---> TF-IDF transform ---> LogReg predict ---> Display
    |
    +--- Summary ------> Pandas aggregation + Embedding centroid ---> Display
    |                     (optional) Flan-T5 generation
    |
    +--- Explanation ---> TF-IDF + LogReg + LIME perturbation ---> Display
    |
    +--- Search -------> MiniLM encode query ---> Cosine sim vs 34K embeddings ---> Display
    |
    +--- RAG ----------> MiniLM retrieve ---> Flan-T5 generate from context ---> Display
    |
    +--- QA -----------> MiniLM retrieve (filtered) ---> Flan-T5 generate ---> Display
```

## Models Used

| Model | Size | Purpose | Loaded when |
|---|---|---|---|
| TF-IDF + LogReg | ~2 MB | Star prediction, topic prediction, LIME | Prediction / Explanation tabs |
| MiniLM (sentence-transformers) | ~120 MB | Query encoding for semantic search | Search / RAG / QA tabs |
| Flan-T5-Small | ~300 MB | Text generation (summaries, RAG, QA) | Summary / RAG / QA tabs |
| Pre-computed embeddings | ~52 MB | 34,428 review vectors (384d) | Always (loaded at startup) |

All models are lazy-loaded (only when the user interacts with the relevant tab) and cached using Streamlit's `@st.cache_resource` decorator, so they load once and persist across interactions.

## Performance Characteristics

- **Prediction tab:** instant (<100ms) — just a TF-IDF transform + matrix multiply
- **Explanation tab:** ~10–15 seconds — LIME generates 300 text perturbations
- **Search tab:** first use ~15s (model load), then <1s per query
- **RAG / QA tabs:** first use ~15s (model load), then 5–10s per query (Flan-T5 generation on CPU)
- **Summary tab:** instant for metrics, 5–10s for AI summary generation
- **Peak memory usage:** ~750 MB above baseline (all models loaded)
