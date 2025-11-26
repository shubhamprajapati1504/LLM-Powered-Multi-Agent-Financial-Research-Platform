from pathlib import Path

# === Where your FAISS store lives (same folder you used on Colab) ===
# Expecting files like: index.faiss, metadata.parquet, documents.parquet (and optionally embeddings.npz)
VECTOR_FAISS_DIR = Path(r"C:\\Users\\harsh\\OneDrive\\Desktop\\LLM Capstone\\Vector Database Store\\vectorstore_faiss")

# === Embedding model (free) ===
EMBEDDING_MODEL  = "BAAI/bge-small-en-v1.5"

TIMEZONE = "Asia/Kolkata"

# === Retrieval defaults ===
TOP_K            = 10

# === Live data defaults ===
NSE_TICKERS = ["HDFCBANK.NS", "ICICIBANK.NS", "RELIANCE.NS", "INFY.NS", "TCS.NS"]
RSS_QUERIES = [
    "RBI repo rate", "NIFTY 50", "HDFC Bank", "ICICI Bank",
    "Reliance Industries", "Infosys", "India CPI inflation", "RBI MPC meeting",
]

# Thesis LLM config (HF Inference API or your own Endpoint)
THESIS_LLM_REPO_ID = "mistralai/Mistral-7B-Instruct-v0.2"   # small, free, works via API

GEMINI_MODEL_ID = "gemini-2.5-flash"

# Prompt generation knobs
THESIS_MAX_NEW_TOKENS = 450
THESIS_TEMPERATURE = 0.2
THESIS_TOP_P = 0.9
THESIS_K_EVIDENCE = 5
THESIS_TOKENS_PER_PASSAGE = 220