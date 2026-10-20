# Comprehensive Study Notes: Machine Learning, Deep Learning & RAG Systems
Author: Akanksha Bharti
Date: 2026

## 1. Transformer Architecture and Self-Attention
The Transformer architecture, introduced by Vaswani et al. in "Attention Is All You Need", replaces recurrence with Multi-Head Self-Attention.
The core formula for Scaled Dot-Product Attention is:
$$Attention(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right)V$$
Where:
- $Q$ is the Query matrix (what we are looking for)
- $K$ is the Key matrix (what each token represents)
- $V$ is the Value matrix (the information content)
- $\sqrt{d_k}$ is the scaling factor preventing vanishing gradients during softmax for high dimensional keys.

Multi-Head Attention allows the model to jointly attend to information from different representation subspaces at different positions.

## 2. Gradient Boosting vs. Random Forest
Both Gradient Boosting (e.g., XGBoost, LightGBM, CatBoost) and Random Forest are ensemble tree methods, but they differ fundamentally in strategy:
- **Random Forest (Bagging)**:
  - Trains multiple deep decision trees in parallel on bootstrap samples of the training data.
  - Reduces **variance** (prevents overfitting).
  - Individual trees are independent and low bias, high variance.
- **Gradient Boosting (Boosting)**:
  - Trains shallow trees (weak learners) sequentially.
  - Each new tree is trained to predict the **pseudo-residuals** (negative gradients of the loss function) of the previous ensemble.
  - Reduces **bias** and variance sequentially.
  - Highly effective for tabular datasets, Kaggle competitions, and fraud detection.

## 3. Retrieval-Augmented Generation (RAG) Architecture
Retrieval-Augmented Generation bridges static parametric knowledge inside Large Language Models (LLMs) with dynamic, private, non-parametric knowledge bases.

### The 4 Pillars of Production RAG:
1. **Document Ingestion & Chunking**:
   - Splitting long texts into overlapping semantic windows (e.g. 500-1000 characters with 10-20% overlap).
   - Preserving source hierarchy: filename, page numbers, headings, section paths.
2. **Dense Vector Embeddings & Vector Stores**:
   - Passing chunks through embedding models (e.g., sentence-transformers, text-embedding-004) to map text to high-dimensional metric space.
   - Indexing via Approximate Nearest Neighbors (ANN) using HNSW (Hierarchical Navigable Small World) in vector stores like ChromaDB or FAISS.
3. **Hybrid Retrieval & Reciprocal Rank Fusion (RRF)**:
   - Combining dense semantic vector search (cosine similarity) with sparse lexical search (BM25 token frequency).
   - RRF formula: $RRF(d) = \sum_{m \in M} \frac{1}{k + r_m(d)}$ where $r_m(d)$ is the rank of document $d$ in system $m$.
4. **Grounded Synthesis & Source Attribution**:
   - Building a constrained prompt containing retrieved passages with citations.
   - Instructing the generator LLM to answer strictly from the context, minimizing hallucinations and enabling auditability.

## 4. Evaluation Metrics for RAG
- **Faithfulness**: Are claims in the generated response strictly grounded in the context?
- **Answer Relevance**: Does the response directly address the user's question?
- **Context Precision**: Did the retriever rank the relevant chunks at the top?
- **Context Recall**: Were all required facts retrieved from the vector store?
