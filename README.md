#  SmartResearch Advisor 🎓

SmartResearch Advisor is an AI-powered tool that helps students generate relevant and personalized research paper topics by analyzing the latest publications from arXiv.



---

## 🚀 Features
-   **Personalized Suggestions:** Generates topics based on the user's chosen domain and academic level (Beginner, Intermediate, Advanced).
-   **Real-time Data:** Fetches the most recent research papers directly from arXiv.
-   **Advanced NLP:** Uses spaCy for text normalization (lemmatization, stopword removal).
-   **Semantic Search:** Employs Sentence-Transformers and FAISS to find the most contextually relevant papers.
-   **Topic Modeling:** Leverages BERTopic to identify and display underlying topic clusters and trends.
-   **Intelligent Ranking:** Ranks suggestions based on a combination of recency, diversity (using MMR), and complexity.
-   **Interactive UI:** Built with Streamlit, featuring an intuitive interface and feedback mechanism.

---

## ⚙️ Technologies Used
-   **Backend:** Python
-   **Web Framework:** Streamlit
-   **NLP:** spaCy, NLTK, Sentence-Transformers (Hugging Face)
-   **Vector Search:** FAISS (Facebook AI Similarity Search)
-   **Topic Modeling:** BERTopic
-   **Containerization:** Docker

---

## 🐳 Docker Quick Start (Recommended)
The easiest way to run the application is with Docker.

1.  **Build the Docker image:**
    ```bash
    docker build -t smart-research-advisor .
    ```

2.  **Run the Docker container:**
    ```bash
    docker run -p 8501:8501 smart-research-advisor
    ```
    You can now access the app at `http://localhost:8501`.

---

## 🛠️ Local Installation
If you prefer to run the application locally:

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/SaniaNasreen/Smart-Research-Advisor.git
    cd Smart-Research-Advisor
    ```

2.  **Create and activate a virtual environment:**
    ```bash
    python -m venv .venv
    ```
    * On **Windows**: `.venv\Scripts\activate`
    * On **macOS/Linux**: `source .venv/bin/activate`

3.  **Install Python dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

4.  **Download the spaCy model:**
    ```bash
    python -m spacy download en_core_web_sm
    ```

## ▶️ Usage
With the virtual environment activated, run the following command to launch the Streamlit app:
```bash
streamlit run smartresearch_app.py
```

---

## ☁️ Deployment
This project is deployment-ready via Docker on platforms like Render, Railway, Fly.io, and Hugging Face Spaces.

- The container listens on `PORT` when provided by the platform.
- Locally it defaults to port `8501`.

Example (platform-managed port):
```bash
docker run -e PORT=7860 -p 7860:7860 smart-research-advisor
```

---

## 📄 License
This project is licensed under the MIT License. See the `LICENSE` file for details.
