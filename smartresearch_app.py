import streamlit as st
import pandas as pd
import re, nltk, time, sqlite3, math
import arxiv
from sentence_transformers import SentenceTransformer
import faiss, numpy as np
from bertopic import BERTopic
from nltk.corpus import stopwords
from datetime import datetime
import torch
import en_core_web_sm

# ---------------------------
# CONFIGURATION
# ---------------------------
CONFIG = {
    "ARXIV_MAX_RESULTS": 150,
    "SEED_HITS": 40,
    "MMR_K": 5,
    "EMBEDDING_MODEL": "sentence-transformers/all-MiniLM-L6-v2",
    "DB_NAME": "feedback.db",
    "CACHE_TTL_SECONDS": 3600 * 6
}

# ---------------------------
# SETUP & MODEL LOADING
# ---------------------------
st.set_page_config(page_title="SmartResearch Advisor", layout="centered")

@st.cache_data
def download_nltk_data():
    nltk.download('punkt')
    nltk.download('stopwords')
    return set(stopwords.words("english"))

STOP = download_nltk_data()

@st.cache_resource
def get_spacy_model():
    return en_core_web_sm.load(disable=['parser', 'ner']) # Disable unused components for speed

@st.cache_resource
def get_embedding_model():
    torch.set_default_dtype(torch.float32)
    return SentenceTransformer(CONFIG["EMBEDDING_MODEL"], device="cpu")

nlp = get_spacy_model()
EMB_MODEL = get_embedding_model()

# ---------------------------
# DATA & PROCESSING FUNCTIONS
# ---------------------------
@st.cache_data(ttl=CONFIG["CACHE_TTL_SECONDS"])
def fetch_arxiv(query="artificial intelligence", max_results=100):
    """Fetches data from arXiv API using the Client for more reliability."""
    try:
        client = arxiv.Client()
        search = arxiv.Search(query=query, max_results=max_results, sort_by=arxiv.SortCriterion.SubmittedDate)
        results = client.results(search)
        rows = []
        for r in results:
            rows.append({
                "title": r.title,
                "abstract": r.summary.replace("\n", " "),
                "url": r.entry_id,
                "published": r.published.date().isoformat()
            })
        if not rows: return pd.DataFrame()
        return pd.DataFrame(rows)
    except Exception as e:
        st.error(f"An error occurred while fetching data from arXiv: {e}")
        return pd.DataFrame()

def prepare_corpus(df: pd.DataFrame) -> pd.DataFrame:
    """Optimized text preparation using nlp.pipe() for batch processing."""
    df = df.copy()
    df["text"] = (df["title"].fillna("") + ". " + df["abstract"].fillna("")).apply(lambda t: re.sub(r"\s+", " ", t).strip())
    
    # Use nlp.pipe for efficient batch processing
    docs = nlp.pipe(df["text"].tolist(), batch_size=50)
    norm_texts = []
    for doc in docs:
        tokens = [
            t.lemma_.lower() for t in doc
            if not t.is_stop and not t.is_punct and not t.like_num and len(t.lemma_) > 2 and t.lemma_.lower() not in STOP
        ]
        norm_texts.append(" ".join(tokens))
    df["norm"] = norm_texts
    return df

# ---------------------------
# VECTOR INDEX & SEARCH
# ---------------------------
class VectorIndex:
    """VectorIndex now expects pre-computed embeddings."""
    def __init__(self, embeddings):
        d = embeddings.shape[1]
        self.index = faiss.IndexFlatIP(d)
        self.index.add(embeddings)

    def search(self, query, k=20):
        q_emb = EMB_MODEL.encode([query], normalize_embeddings=True, convert_to_numpy=True)
        _, ids = self.index.search(q_emb, k)
        return [int(i) for i in ids[0]]

# ---------------------------
# TOPIC MODELING & RANKING
# ---------------------------
def generate_topic_clusters(norm_texts, embeddings):
    """Optimized to accept pre-computed embeddings."""
    topic_model = BERTopic(min_topic_size=10, calculate_probabilities=False, verbose=False, embedding_model=EMB_MODEL)
    topics, _ = topic_model.fit_transform(norm_texts, embeddings=embeddings)
    info = topic_model.get_topic_info()
    suggestions = []
    for _, row in info.head(8).iterrows():
        if row.Topic == -1: continue
        words = [w for w, _ in topic_model.get_topic(row.Topic) or []][:5]
        if words:
            suggestions.append("Cluster: " + ", ".join(words))
    return suggestions

def recency_weight(pub_date, tau=180):
    try:
        days_old = (datetime.now().date() - datetime.fromisoformat(pub_date).date()).days
        return math.exp(-days_old / tau)
    except: return 1.0

def mmr_select(candidates_emb, query_emb, k=5, lambda_=0.7):
    # (Function remains the same)
    chosen_indices, candidate_indices = [], list(range(len(candidates_emb)))
    sim_to_query = (candidates_emb @ query_emb.T).flatten()
    while candidate_indices and len(chosen_indices) < k:
        mmr_scores = []
        for i in candidate_indices:
            sim_to_chosen = 0
            if chosen_indices: sim_to_chosen = np.max(candidates_emb[i] @ candidates_emb[chosen_indices].T)
            mmr = lambda_ * sim_to_query[i] - (1 - lambda_) * sim_to_chosen
            mmr_scores.append((mmr, i))
        mmr_scores.sort(reverse=True, key=lambda x: x[0])
        best_index = mmr_scores[0][1]
        chosen_indices.append(best_index)
        candidate_indices.remove(best_index)
    return chosen_indices

def rank_topics(seed_df, seed_embeddings, query, level, k=5, lambda_=0.7):
    q_emb = EMB_MODEL.encode([query], normalize_embeddings=True, convert_to_numpy=True)
    top_ids = mmr_select(seed_embeddings, q_emb, k=k, lambda_=lambda_)
    ranked = seed_df.iloc[top_ids].copy()
    ranked["recency"] = ranked["published"].apply(recency_weight)
    if level == "Beginner": ranked["score"] = ranked["recency"] * ranked["abstract"].apply(lambda x: 1 / (1 + len(x.split())))
    elif level == "Intermediate": ranked["score"] = ranked["recency"] * 1.0
    else: ranked["score"] = ranked["recency"] * ranked["abstract"].apply(lambda x: len(x.split()))
    return ranked.sort_values("score", ascending=False)

# ---------------------------
# CACHED DATA PIPELINE
# ---------------------------
@st.cache_data(ttl=CONFIG["CACHE_TTL_SECONDS"])
def load_and_process_data(domain):
    """A single cached function to handle all heavy lifting."""
    df = fetch_arxiv(query=domain, max_results=CONFIG["ARXIV_MAX_RESULTS"])
    if df.empty: return None, None, None
    df = prepare_corpus(df)
    embeddings = EMB_MODEL.encode(df["norm"].tolist(), normalize_embeddings=True, convert_to_numpy=True)
    vector_index = VectorIndex(embeddings)
    return df, vector_index, embeddings

# ---------------------------
# FEEDBACK DATABASE
# ---------------------------
def init_db():
    with sqlite3.connect(CONFIG["DB_NAME"]) as con: con.execute("CREATE TABLE IF NOT EXISTS feedback(ts INTEGER, domain TEXT, level TEXT, title TEXT, vote INTEGER)")
def save_feedback(domain, level, title, vote):
    with sqlite3.connect(CONFIG["DB_NAME"]) as con: con.execute("INSERT INTO feedback VALUES(?,?,?,?,?)", (int(time.time()), domain, level, title, vote))
init_db()

# ---------------------------
# STREAMLIT UI
# ---------------------------
def main():
    st.title("🎓 SmartResearch Advisor")
    st.write("AI-powered research topic generator for students, using the latest papers from arXiv.")

    domain = st.selectbox("Choose a research domain", ["Artificial Intelligence", "Web Development", "Data Science", "Cybersecurity", "Blockchain"])
    level = st.selectbox("Select your academic level", ["Beginner", "Intermediate", "Advanced"])

    with st.expander("Advanced Settings"):
        num_topics = st.slider("Number of topics to generate", 3, 10, CONFIG["MMR_K"])
        lambda_param = st.slider("Diversity vs. Relevance (Lambda for MMR)", 0.0, 1.0, 0.7, help="Lower value for more diverse topics, higher value for more relevance to the query.")

    if st.button("✨ Generate Topics"):
        with st.spinner("Running analysis... This may take a moment on the first run for a new domain."):
            # Call the single cached function
            df, vector_index, embeddings = load_and_process_data(domain)
            
            if df is None:
                st.error("Could not find any papers for this domain. Please try another one.")
                st.stop()

            search_query = f"{domain} {level} research for students"
            hit_ids = vector_index.search(search_query, k=CONFIG["SEED_HITS"])
            
            seed_df = df.iloc[hit_ids]
            seed_embeddings = embeddings[hit_ids]
            
            topics = generate_topic_clusters(seed_df["norm"].tolist(), seed_embeddings)
            ranked = rank_topics(seed_df, seed_embeddings, search_query, level, k=num_topics, lambda_=lambda_param)

        st.success("✅ Analysis Complete!")
        # (The rest of the UI for displaying results remains the same)
        st.subheader("✅ Top Suggested Topics")
        if ranked.empty: st.warning("No suitable topics could be ranked based on the criteria.")
        else:
            for i, row in ranked.iterrows():
                with st.container(border=True):
                    st.markdown(f"#### {row['title']}")
                    st.caption(f"Published: {row['published']} | [Read Paper]({row['url']})")
                    st.write(row['abstract'][:350] + "...")
                    feedback_key_prefix, (col1, col2, _) = f"fb_{i}", st.columns([1,1,5])
                    if col1.button("👍", key=f"{feedback_key_prefix}_up", help="Good suggestion"):
                        save_feedback(domain, level, row['title'], 1); st.toast("Feedback recorded!")
                    if col2.button("👎", key=f"{feedback_key_prefix}_down", help="Bad suggestion"):
                        save_feedback(domain, level, row['title'], -1); st.toast("Feedback recorded!")
        
        st.subheader("🔎 Related Topic Clusters")
        if not topics: st.info("No distinct topic clusters were identified.")
        else:
            st.info("These are common themes found within the relevant papers, generated by BERTopic.")
            for t in topics: st.markdown(f"- {t}")
        if not ranked.empty: st.download_button("📥 Download Ranked Topics (CSV)", ranked.to_csv(index=False), f"ranked_topics_{domain.lower().replace(' ','_')}.csv", "text/csv")

if __name__ == "__main__":
    main()
