# LLM Capstone Investment Analyst

A production-style multi-agent investment analysis framework for Indian equities. This project combines market data ingestion, technical analysis, evidence-backed thesis generation, verification, and structured reporting in a reproducible pipeline.

---

## 🚀 What This Project Does

- Ingests stock market data and company fundamentals using `yfinance`
- Builds a data-driven investment analysis workflow with dedicated agents
- Uses semantic search and citation-backed evidence to support investment theses
- Applies rigorous verification to detect hallucinations and check internal consistency
- Produces structured output that can be consumed by dashboards, APIs, or reports

---

## 🧠 Architecture Overview

```text
User Query
   │
   ▼
Router Agent
   │
   ├─> Data Agent
   │     • Price + fundamentals
   │     • FAISS evidence search
   │
   ├─> Analyst Agent
   │     • Technical indicators
   │     • Risk metrics
   │
   ├─> Thesis Agent
   │     • Bull/bear thesis generation
   │     • Evidence citation
   │
   ├─> Verification Agent
   │     • Claim validation
   │     • Consistency checks
   │
   └─> Final Summary Agent
         • Structured recommendation
         • Reporting output
```

---

## 📁 Key Files

- `crew_ai.py` – Crew-based stock analysis flow using custom tools and agents
- `app.py` – Entrypoint for the graph-driven workflow pipeline
- `streamlit_run.py` – Streamlit demo launcher for interactive queries
- `agents/` – Specialized agent modules:
  - `router_agent.py`
  - `data_agent.py`
  - `analyst_agent.py`
  - `thesis_agent.py`
  - `verification_agent.py`
  - `final_summary.py`
  - `portfolio_agent.py`
  - `simulation_agent.py`
  - `index_loader.py`
- `requirements.txt` – Python dependencies for the full stack

---

## 🧩 Agent Responsibilities

### Router Agent
- Classifies user intent
- Extracts stock or portfolio entities
- Routes queries into the correct analysis pipeline

### Data Agent
- Fetches historical price data and fundamentals from `yfinance`
- Builds evidence context using FAISS semantic search
- Normalizes and structures time-series and news inputs

### Analyst Agent
- Computes technical indicators like SMA, RSI, volatility, and drawdown
- Evaluates fundamentals such as market cap, P/E, P/B, and dividend yield
- Produces a quantitative snapshot for thesis generation

### Thesis Agent
- Generates bull and bear arguments
- Anchors each claim with numbered evidence citations
- Limits output to the provided metrics and retrieved sources

### Verification Agent
- Detects unsupported numerical claims
- Validates internal consistency of the final thesis
- Flags missing citations or evidence gaps

### Final Summary Agent
- Produces a clean recommendation report
- Formats output for human readers and structured consumers
- Emphasizes the final decision, risk assessment, and supporting rationale

---

## 🛠️ Setup Instructions

### 1. Clone and open the repository

```bash
git clone <your-repo-url>
cd LLM-Capstone-Project-main
```

### 2. Create a virtual environment

```bash
python -m venv .venv
.venv\Scripts\activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Set required API keys

This project uses Google Generative Language and may require environment variables or local authentication configuration for the chosen LLM provider.

Example:

```powershell
setx GOOGLE_API_KEY "your_api_key_here"
```

> If you use a different provider, configure the equivalent credentials in your environment.

---

## ▶️ Run Examples

### Run the pipeline

```bash
python app.py
```

### Run the Crew AI stock analysis

```bash
python crew_ai.py
```

### Run the Streamlit demo

```bash
python streamlit_run.py
```

---

## ✅ Why This README Matters

A strong README is the first credibility signal for a public demo. It should clearly explain:
- what the system does,
- how it is built,
- how to run it,
- and what each agent contributes.

If you want a polished presentation link, make sure the GitHub repository is set to **public** and this README is available at the root.

---

## 📌 Notes for Publication

- Host the repository publicly on GitHub for maximum trust.
- Keep the README updated when agents or data sources change.
- Add a screenshot or demo GIF later to improve the project preview.

---

## 📚 Suggested Next Enhancements

- Add a `demo/` folder with sample output JSON reports
- Add `docs/` diagrams or an architecture whitepaper
- Add automated tests for agent outputs and data validation
- Add a `CONTRIBUTING.md` for collaborators
