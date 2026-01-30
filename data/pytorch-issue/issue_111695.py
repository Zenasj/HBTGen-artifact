from typing import List
import torch
from langchain.embeddings import HuggingFaceEmbeddings

class EmbeddingsModel:
    def __init__(self):
        self.model_name = "sentence-transformers/multi-qa-MiniLM-L6-cos-v1"
        encode_kwargs = {"normalize_embeddings": True}

        if torch.cuda.is_available():
            # Use CUDA GPU
            device = torch.device("cuda:0")
        elif torch.backends.mps.is_available():
            # Use Apple M1 Metal Acceleration
            device = torch.device("mps")
        else:
            device = torch.device("cpu")

        self.device = device
        model_kwargs = {"device": device}
        self.embeddings_model = HuggingFaceEmbeddings(
            model_name=self.model_name, encode_kwargs=encode_kwargs, model_kwargs=model_kwargs
        )

    def embed_documents(self, docs: List[str]):
        print(f"Using device: {self.device} to embed {len(docs)} documents")
        return self.embeddings_model.embed_documents(docs)

model = EmbeddingsModel()
embeddings = model.embed_documents(["this is a document", "so is this"])
print(f"Created embeddings of length {len(embeddings)}")