# RAG_Against_The_Machine

*This project has been created as part of the 42 curriculum by tmoga*

"Heyo it's just another bombtrack!" - RAG de la Rocha certainly... 

## Description

...

## Limits

Qwen/Qwen3-0.6B limitations de raisonnement (mauvaise orthographe, style raide, hallucinations...)

all-MiniLM-L6-v2 limitations to 256 tokens, ce qui est idyllique pour des chunks de 2000 caracteres.
Switch to Qwen/Qwen3-Embedding-8B. Tres mauvaise idee car son initialisation prend 15,1GB et affiche un message d'erreur à ce sujet : "MPS backend out of memory (MPS allocated: 19.44 GiB, other allocations: 752.00 KiB, max allowed: 20.13 GiB). Tried to allocate 438.00 MiB on shared pool."

Une autre limite du RAG dans cette exercice est que le retrieval se contente des fichiers .py et .md du corpus, alors que beaucoup d'autres types de fichiers sont utilises et pourraient servir a augmenter la generation du LLM.

## Instructions

System Architecture

... 

Chunking strategy

Utilisation de la librairie Langchain 

Utilisation de MarkdownTextSplitter au lieu de MarkdownHeaderTextSplitter. Choix effectué en fonction de la contrainte max_chunk_size pour respecter l'énoncé et pouvoir conserver les index dans le texte d'origine. Mais en terme sémantique, l'utilisation de MarkdownHeaderTextSplitter aurait ete meilleure pour garder un meilleur contexte sur les chunks.

Retrieval method

Retrieval semantique pertinent 

Performance analysis

Le retrieval demandé par le sujet basé sur l'algorithme BM25 (lexical) est plutôt efficace et a permis d'atteindre des scores initiaux de 82% pour les docs et 58% pour le code sans ajouter de stemmer ou autres optimisations. Ceci etant dit, je pensais qu'une approche par retrieval sémantique pouvait améliorer ces scores. En utilisant le modele d'embedding "sentence-transformers/all-MiniLM-L6-v2" je n'ai pas réussi à obtenir de bons scores séparément (55% docs et 37,4% code) en raison des limitations de tokens du modele (256 tokens) qui ne permettaient pas de vectoriser l'ensemble du chunk qui par defaut fait 2000 caracteres.

Les performances du retrieval sont bonnes et l'entiereté de la pipeline met moins d'une minute à se lancer. Les limites de performances vont se situer du coté de la génération de réponse qui est tres longue, en particulier sur de grands batch de questions. La methode answer_dataset() prend environ 25 minutes.

Design decisions:

J'ai décidé pour ce projet de refuser toute approche déterministe lors de la recherche des sources afin d'être au plus proche d'un RAG en production.
En effet, la méthode search_dataset() pourrait décider de ne citer en source que des fichiers .md lorsqu'on lui donne un chemin où le fichier .json mentionne "_docs_" (*ex: datasets_public/public/UnansweredQuestions/dataset_docs_public.json*). Mais cela irait à l'opposé de l'idée même d'un vrai RAG qui en pratique ne peut pas initialement savoir quel type de fichier sera le mieux adapté pour répondre à une question (car c'est le but du retriever).
C'est pour cela que mon indexing ne sépare pas les sources .md et .py mais calcule un score pour l'ensemble des sources, indépendamment de leur nature en évitant ensuite un duplicata inutile d'une pipeline qui augmenterait la complexité spatiale du programme.

Le problème est que la recherche lexicale seule ne suffit pas à avoir un score recall@k correct. 

Flexible recall@k pour l'evaluation qui scale sur la longueur des "retrieved_sources" du dataset généré par search_dataset()


Challenges faced

Determiner la longueur max_new_tokens pour la generation pour pas perdre en perf et en meme temps avoir des resultats coherent;
Laisser le processus de pensee <think> de Qwen pour gagner en precision;
Determiner un bon prompting (10 prompts differents essayés) pour les contraintes faithful, source-grounded etc. Qwen est capricieux.
Une balise dans le prompt est commentée pour une meilleure generation, mais en la reactivant Qwen generera les sources avec une precision approximative. 

Benchmark max token:

Code :

- 650 tokens    7/10 (3 tronqués)    167.8
- 500 tokens    7/10                 156.1
- 400 tokens    7/10                 143
- 300 tokens    6/10                 134

Docs :

- 650 tokens    10/10   87.3
- 500 tokens    10/10   86.1
- 400 tokens    10/10   85.2
- 300 tokens    9/10    81.4

Accessible dans utils.thinking_length()

Example Usage

index           : uv run python -m src index
search          : uv run python -m src search "How to configure the OpenAI server?" --k 5
search dataset  : uv run python -m src search_dataset --dataset_path datasets_public/public/UnansweredQuestions/dataset_code_public.json
moulinette      : ./moulinette evaluate_student_search_results output/search_results/UnansweredQuestions/dataset_docs_public.json datasets_public/public/AnsweredQuestions/dataset_docs_public.json --k 10 --max_context_length 2000

answer_dataset  : uv run python -m src answer_dataset
                  --student_search_results_path
                  data/output/search_results/UnansweredQuestions/
                  dataset_docs_public.json
                  --save_directory
                  data/output/search_results_and_answer/
                  UnansweredQuestions

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
