from sentence_transformers import SentenceTransformer

model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2", device="mps")
model = model.half() # problem

print(model.encode("some text").sum())