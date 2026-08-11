SRC = src
VENV = .venv
PYTHON = $(VENV)/bin/python3
RUN = uv run
TRASH = $(VENV) .mypy_cache .uv_cache data/processed data/chunking_output output $(SRC)/__pycache__ __pycache__ $(VENV) .mypy_cache data/output .uv_cache data/output
MYPY_FLAGS = --warn-return-any --warn-unused-ignores --ignore-missing-imports \
			--disallow-untyped-defs --check-untyped-defs
UV_CACHE = .uv_cache
HF_CACHE = .hf_cache
UV_CACHE_DIR = $(CURDIR)/$(UV_CACHE)
HF_CACHE_DIR = $(CURDIR)/$(HF_CACHE)

UV_SETUP = UV_CACHE_DIR="$(UV_CACHE_DIR)" UV_LINK_MODE=copy
HF_SETUP = HF_HOME="$(HF_CACHE_DIR)" HUGGINGFACE_HUB_CACHE="$(HF_CACHE_DIR)/hub" TRANSFORMERS_CACHE="$(HF_CACHE_DIR)/transformers"


# ANSI codes
GREEN = \033[0;32m
RED = \033[0;31m
YELLOW = \033[0;33m
BOLD = \033[1m

SUCCESS = \033[1;92m
RESET = \033[0m


install:
	@echo "$(BOLD)Initializing project environment:$(RESET)"
	@echo "$(YELLOW)Creating virtual environment...$(RESET)"
	uv venv $(VENV) --clear > /dev/null 2>&1
	@echo "$(GREEN)Virtual environment setup complete$(RESET)"

	@bash -c '\
		$(UV_SETUP) $(HF_SETUP) UV_PROJECT_ENVIRONMENT="$(VENV)" \
		uv sync & pid=$!; \
		spin="-\|/"; i=0; \
		yellow="\033[0;33m"; reset="\033[0m"; \
		while kill -0 $$pid 2>/dev/null; do \
			c=$$(echo "$$spin" | cut -c$$((i % 4 + 1))); \
			printf "\r$${yellow}[%s] Installing dependencies...$${reset}" "$$c"; \
			i=$$((i + 1)); \
			sleep 0.1; \
		done; \
		wait $$pid; exit $$?'

	@echo "\n$(GREEN)All dependencies successfully installed$(RESET)"
	@echo "\n$(SUCCESS)Project can now be run safely!$(RESET)"


run:
	@$(HF_SETUP) python -m $(SRC) $(ARGS)

lint:
	@echo "$(BOLD)Verifying project source code:$(RESET)"

	@echo "$(YELLOW)Running flake8 and mypy...$(RESET)"
	@$(RUN) -m flake8 .
	@$(RUN) -m mypy . $(MYPY_FLAGS)
	@echo "$(GREEN)LINT OK$(RESET)"

debug:
	@$(PYTHON) -m pdb -m $(SRC)

clean:
	@echo "$(YELLOW)Initiating cleanup operation...$(RESET)"
	@found=0; \
	for i in $(TRASH); do \
		if [ -d "$$i" ]; then \
			echo " - Removed directory $(RED)'$$i'$(RESET)"; \
			found=1; \
		elif [ -f "$$i" ]; then \
			echo " - Removed file $(RED)'$$i'$(RESET)"; \
			found=1; \
		fi; \
	done; \
	if [ $$found -eq 0 ]; then \
		echo "$(GREEN)Nothing to clean :)$(RESET)"; \
	fi
	@find . -type d -name "__pycache__" -print -exec rm -rf {} +
	@rm -rf $(TRASH)
	@echo "$(SUCCESS)\nCleanup complete!$(RESET)"

.PHONY: install run lint debug clean
