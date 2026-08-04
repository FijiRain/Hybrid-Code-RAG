import fire
from rich import print
from .cli import Cli


def main() -> None:
    try:
        fire.Fire(Cli)
    except Exception as e:
        print(f"[bold red]ERROR - {e.__class__.__name__} :[/bold red]\n"
              f"[yellow]{e}[/yellow]")


if __name__ == "__main__":
    main()
