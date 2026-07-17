import fire
from .cli import Cli


def main() -> None:
    try:
        fire.Fire(Cli)
    except Exception as e:
        print(f"ERROR - {e.__class__.__name__} :\n{e}")


if __name__ == "__main__":
    main()
