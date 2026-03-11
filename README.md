# KYC Adverse Agent

An AI-powered adverse media screening system that automatically searches, scrapes, analyzes, and evaluates public information about individuals to identify potential financial crime risks such as fraud, corruption, or sanctions exposure.

---

## Technology Stack

- **Orchestration:** Python-based modular pipeline  
- **API Layer:** FastAPI  
- **LLM Runtime / Models:** Ollama (Phi-3 Mini, Mistral 7B)  
- **Embedding Model:** Sentence Transformers (all-MiniLM-L6-v2)  
- **Search & Data Collection:** Multi-engine web search + custom web scraping  
- **AI Processing:** Semantic embeddings, cosine similarity ranking, sentence-level filtering  
- **UI Layer:** Streamlit  
- **Language:** Python 3.11  
- **Containerization:** Docker  
- **CI/CD:** GitHub Actions  
- **Testing:** Pytest  

---

## AI & System Design Patterns

This project demonstrates several modern **AI engineering and system architecture patterns**:

- **Sentence-level semantic filtering** to reduce LLM input size and improve inference efficiency  
- **Embedding-based similarity ranking** using transformer embeddings and cosine similarity  
- **Agent-based modular architecture** separating search, scraping, risk analysis, and reporting components  
- **Multi-model inference pipeline** combining a smaller compression model and a larger reasoning model  
- **Observability-first design** with detailed logging across each pipeline step  
- **API-first microservice architecture** enabling integration with external systems  
- **Containerized deployment architecture** for reproducibility and portability  

---

## Capabilities Demonstrated

This project showcases several **production-oriented AI engineering skills**:

- Designing an **end-to-end LLM pipeline** for real-world financial risk analysis  
- Implementing **semantic search and evidence ranking using transformer embeddings**  
- Building **agent-style modular AI services** for search, scraping, risk evaluation, and reporting  
- Optimizing LLM workflows using **sentence filtering and prompt compression**  
- Creating **observable AI systems** with step-by-step telemetry and performance logs  
- Building **API-driven AI services** suitable for integration with enterprise platforms  
- Developing **containerized AI services** using Docker  
- Implementing **CI pipelines using GitHub Actions** for automated testing and builds  

---

## High-Level Flow

User → Streamlit UI → FastAPI API → Search Agent → Scraper → Embedding & Ranking → Evidence Compression (Phi-3) → Reasoning Model (Mistral) → Risk Report

---

## Future Plans

- **Multi-agent architecture** with specialized agents for sanctions, fraud, and corruption analysis  
- **Vector database integration** for persistent knowledge retrieval and historical case analysis  
- **Improved scraping reliability** using headless browser automation  
- **Structured risk scoring models** combining LLM reasoning with rule-based signals  
- **Enterprise integrations** with compliance systems and KYC platforms  

---
