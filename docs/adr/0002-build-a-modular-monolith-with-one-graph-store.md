---
status: accepted
---

# Build a modular monolith with one graph store

AH-MemoryHub will ship as one Python application with internal domain modules, one browser frontend, and Neo4j Community as its only server-side datastore. Separate GraphRAG, vector-database, engine, and ingestion services were rejected because the hackathon scale does not justify synchronising multiple stores or deployment units; AH Core remains the only mutation boundary, and the vanilla-RAG baseline reuses the same chunks and embeddings without becoming a second source of truth.
