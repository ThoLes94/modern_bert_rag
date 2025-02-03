import argparse
from pathlib import Path

from html2text import HTML2Text


def convert_to_txt(file_dir: str, h: HTML2Text) -> None:
    list_file = list(Path(file_dir).rglob("*.html"))
    for file in list_file:
        with open(file) as f:
            txt = f.read()

        filtered_text = h.handle(txt)
        with open(str(file).replace(".html", ".txt"), "w") as f:
            f.write(filtered_text)


if __name__ == "__main__":
    h = HTML2Text()
    h.ignore_links = True
    parser = argparse.ArgumentParser()

    parser.add_argument("corpus-path", type=str, help="Path to the corpus to filter")
    arg = parser.parse_args()
    convert_to_txt(arg.corpus_path, h)
