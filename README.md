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

Design decisions:

J'ai décidé pour ce projet de refuser toute approche déterministe lors de la recherche des sources afin d'être au plus proche d'un RAG en production.
En effet, la méthode search_dataset() pourrait décider de ne citer en source que des fichiers .md lorsqu'on lui donne un chemin où le fichier .json mentionne "_docs_" (*ex: datasets_public/public/UnansweredQuestions/dataset_docs_public.json*). Mais cela irait à l'opposé de l'idée même d'un vrai RAG qui en pratique ne peut pas initialement savoir quel type de fichier sera le mieux adapté pour répondre à une question (car c'est le but du retriever).
C'est pour cela que mon indexing ne sépare pas les sources .md et .py mais calcule un score pour l'ensemble des sources, indépendamment de leur nature en évitant ensuite un duplicata inutile d'une pipeline qui augmenterait la complexité spatiale du programme.

Le problème est que la recherche lexicale seule ne suffit pas à avoir un score recall@k correct. 


Challenges faced
Example Usage

## Resources 

- RAG Course for Beginners [video](https://www.youtube.com/watch?v=swvzKSOEluc)
- RAG Explained [video](https://www.youtube.com/watch?v=dDkynerzV-Q)
- "RAG expliqué simplement" [video](https://www.youtube.com/watch?v=5V3t7YXYjZc)
- Vectors dot product [lesson](https://www.maths-et-tiques.fr/telech/ProduitScal.pdf)
- What is a [vLLM](https://www.redhat.com/en/topics/ai/what-is-vllm)?
- TF-IDF [video](https://www.youtube.com/watch?v=OymqCnh-APA)
- BM25 Algorithm [article](https://www.elastic.co/fr/blog/practical-bm25-part-2-the-bm25-algorithm-and-its-variables)
- How BM25 works in [practice](https://www.youtube.com/watch?v=TW9vHU1GpU4)
- Another video on [BM25](https://www.youtube.com/watch?v=ruBm9WywevM&list=WL&index=87)
- How to chunk Python and .md [video](https://www.youtube.com/watch?v=aZHXY_oVQt8)
- LangChain [doc](https://lagnchain.readthedocs.io/en/stable/modules/indexes) on splitters
- BM25S [doc](https://bm25s.github.io)
- HF Qwen3-0.6B [doc](https://huggingface.co/Qwen/Qwen3-0.6B)
