from dotenv import load_dotenv

load_dotenv()

import os

import openai

if os.environ.get("OPENAI_API_KEY") == None:
    raise ValueError("OPENAI_API_KEY environment variable not set")


class OpenAIAdaEmbedder:
    def __init__(self):
        self.embedder = openai.Embedding

    def calculate_embeddings(self, text_inputs: list) -> list:
        response = self.embedder.create(input=text_inputs, model="text-embedding-ada-002")
        embeddings = [data["embedding"] for data in response["data"]]

        return embeddings


if __name__ == "__main__":
    embedder = OpenAIAdaEmbedder()
    import numpy as np

    print(np.max(embedder.calculate_embeddings(["hello"] * 5), axis=1))
    import time

    time.sleep(10)
    print(np.max(embedder.calculate_embeddings(["hello"] * 5), axis=1))
