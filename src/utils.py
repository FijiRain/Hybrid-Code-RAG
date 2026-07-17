from pathlib import Path


def load_files(dir: str) -> list[tuple[str, str]]:
    res = []
    dir_path = Path("./data/raw/vllm-0.10.1")
    extension = {".py", ".md"}

    selected_files = [str(f) for f in dir_path.rglob("*")
                      if f.suffix in extension]
    for file in selected_files:
        with open(file, "r") as content:
            res.append((file, content.read()))
    return res
