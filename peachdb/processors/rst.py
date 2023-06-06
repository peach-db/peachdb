import os
from typing import List

import modal
import pandas as pd
from bs4 import BeautifulSoup, NavigableString


def get_rst_files(dir_path: str):
    if not os.path.isdir(dir_path):
        raise ValueError(f"{dir_path} is not a directory")

    rst_files = []

    for root, dirs, files in os.walk(dir_path):
        for name in files:
            if name.endswith(".rst"):
                rst_files.append(os.path.join(root, name))

    return rst_files


image = modal.Image.debian_slim().apt_install("pandoc").pip_install("pypandoc", "beautifulsoup4", "pandas")
stub = modal.Stub("pandoc-rst-parser")

if stub.is_inside():
    import pypandoc  # type: ignore


@stub.function(image=image)
def get_sections_and_convert_to_html(rst_text: str) -> List[str]:
    html = pypandoc.convert_text(rst_text, "html", format="rst")
    soup = BeautifulSoup(html, "html.parser")

    markdown = []

    h2_tags = soup.find_all("h2")

    for tag in h2_tags:
        html_content = tag.prettify()

        next_sibling = tag.find_next_sibling()
        while next_sibling and next_sibling.name != "h2":
            if isinstance(next_sibling, NavigableString):
                html_content += next_sibling.strip()
            else:
                html_content += next_sibling.prettify()
            next_sibling = next_sibling.find_next_sibling()

        # Convert the HTML content to Markdown
        markdown.append(pypandoc.convert_text(html_content, "md", format="html"))

    return markdown


def create_csv(
    folder_path: str,
    output_csv_path: str,
):
    """
    Takes a path to a directory containing rst files and segments them into sections, and converts them into Markdown.
    For upstream use into the peachdb pipeline.

    """
    file_names = get_rst_files(folder_path)
    print(f"Processing {len(file_names)} files.")
    file_contents = []
    for rst_file in file_names:
        with open(rst_file, "r") as f:
            file_contents.append(f.read())

    dataset_fnames = []
    dataset_txts = []
    dataset_ids: List[int] = []

    with stub.run():
        parsed_rsts = list(get_sections_and_convert_to_html.map(file_contents))
        for fname, parsed_rst_sections in zip(file_names, parsed_rsts):
            for section in parsed_rst_sections:
                dataset_fnames.append(fname)
                dataset_txts.append(section)
                dataset_ids.append(dataset_ids[-1] + 1 if len(dataset_ids) > 0 else 0)

    pd.DataFrame({"file_name": dataset_fnames, "text": dataset_txts, "ids": dataset_ids}).to_csv(output_csv_path)


if __name__ == "__main__":
    create_csv("/home/ec2-user/solidity/docs", "docs_segmented.csv")
