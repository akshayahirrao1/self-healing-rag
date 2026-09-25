# Self-Healing RAG: Full-Stack Upgrade Roadmap

This document outlines the step-by-step plan to upgrade the Self-Healing RAG pipeline from a CLI script into an impressive, production-ready full-stack application suitable for a resume portfolio.

## 🎯 Goal
Transform the project into a visually stunning, interactive web application that explicitly demonstrates the AI's "self-healing" reasoning process to the end user.

---

## 🛠️ Step 1: FastAPI Backend
**Objective**: Expose the LangGraph workflow via a REST API so a frontend can interact with it.
- Create `src/api.py` using FastAPI.
- Implement an endpoint (e.g., `/api/chat`) that accepts user queries.
- **Crucial feature**: The endpoint must stream the intermediate LangGraph state updates back to the client (e.g., using Server-Sent Events). This allows the frontend to know exactly when the AI is retrieving, generating, evaluating, and reformulating.

## 🎨 Step 2: Beautiful React Chat UI (Vite + React)
**Objective**: Build a premium, modern chat interface that "WOWs" the user.
- Scaffold a React app using Vite (e.g., `frontend/`).
- Use Vanilla CSS or TailwindCSS for a sleek dark-mode aesthetic with smooth gradients and glassmorphism.
- **Crucial feature**: Build a "Reasoning visualizer" component. Underneath the AI's chat bubble, show micro-animations of its current state:
  - 🔄 *Retrieving documents...*
  - ❌ *Critic rejected draft: "No mention of sick leave."*
  - 🔄 *Reformulating search query...*
  - ✅ *Answer grounded!*

## 📁 Step 3: Dynamic Document Upload
**Objective**: Allow users to dynamically add knowledge to the RAG system from the UI.
- Add a "Drag & Drop" file upload zone to the React UI.
- Add an `/api/upload` endpoint to the FastAPI backend.
- When a user uploads a PDF or TXT file, process it using the existing `ingestion.py` logic (chunking + embedding) and insert it into the ChromaDB dynamically.

## 🧠 Step 4: Advanced RAG (Reranking)
**Objective**: Show advanced AI engineering skills by improving retrieval accuracy.
- Implement a **Cross-Encoder Reranker** in the retrieval node.
- Flow: Fetch top 10-15 chunks from Chroma -> Rerank them based on the specific query -> Keep the top 3-5 most relevant chunks.
- Update the documentation to highlight this improvement in precision.

## 🐳 Step 5: Dockerization
**Objective**: Prove deployment capability and make the project "One-Click Run" for recruiters.
- Write a `Dockerfile` for the FastAPI backend.
- Write a `Dockerfile` for the React frontend (or serve the built React static files from FastAPI).
- Write a `docker-compose.yml` at the project root.
- A recruiter should be able to run `docker-compose up -d` and immediately access the beautiful UI on their `localhost`.

---

### Status Tracking
- [ ] Step 1: FastAPI Backend
- [ ] Step 2: Beautiful React Chat UI
- [ ] Step 3: Dynamic Document Upload
- [ ] Step 4: Advanced RAG (Reranking)
- [ ] Step 5: Dockerization
