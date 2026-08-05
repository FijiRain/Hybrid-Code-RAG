# Dockerfile pour faire tourner la moulinette 42 (nécessite Linux x86-64)
# À builder avec --platform linux/amd64 sur Mac Apple Silicon

FROM --platform=linux/amd64 ubuntu:22.04

# Évite les prompts interactifs pendant apt install
ENV DEBIAN_FRONTEND=noninteractive

# Outils de base + dépendances utiles pour uv et pour compiler certains paquets Python
RUN apt update && apt install -y \
    curl \
    git \
    ca-certificates \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Installation de uv
RUN curl -LsSf https://astral.sh/uv/install.sh | sh
ENV PATH="/root/.local/bin:$PATH"

# Dossier de travail dans le conteneur
WORKDIR /app

# On copie d'abord les fichiers de dépendances pour profiter du cache Docker
COPY pyproject.toml uv.lock* ./

# Installe les dépendances (si pas de uv.lock, uv sync le crée)
RUN uv sync --no-install-project || true

# Copie le reste du projet
COPY . .

# Sync final (au cas où le code du projet lui-même doit être installé)
RUN uv sync

CMD ["bash"]
