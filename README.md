# RAG_Against_The_Machine

*This project has been created as part of the 42 curriculum by tmoga*

## Description

## Limits

Chunking: Utilisation de MarkdownTextSplitter au lieu de MarkdownHeaderTextSplitter. Choix effectué en fonction de la contrainte max_chunk_size pour respecter l'énoncé et pouvoir conserver les index dans le texte d'origine. Mais en terme sémantique, l'utilisation de MarkdownHeaderTextSplitter aurait ete meilleure pour garder un meilleur contexte sur les chunks.

## Instructions

System Architecture
Chunking strategy
Retrieval method
Performance analysis
Design decisions
Challenges faced
Example Usage



## Resources 

- RAG Course for Beginners [video](https://www.youtube.com/watch?v=swvzKSOEluc)
- RAG Explained [video](https://www.youtube.com/watch?v=dDkynerzV-Q)
- RAG expliqué simplement [video](https://www.youtube.com/watch?v=5V3t7YXYjZc)
- Cours [produit scalaire](https://www.maths-et-tiques.fr/telech/ProduitScal.pdf)
- What is a [vLLM](https://www.redhat.com/en/topics/ai/what-is-vllm)?
- BM25 Algorithm [article](https://www.elastic.co/fr/blog/practical-bm25-part-2-the-bm25-algorithm-and-its-variables)
- How to chunk Python and .md [video](https://www.youtube.com/watch?v=aZHXY_oVQt8)
- LangChain [doc](https://lagnchain.readthedocs.io/en/stable/modules/indexes) on splitters
