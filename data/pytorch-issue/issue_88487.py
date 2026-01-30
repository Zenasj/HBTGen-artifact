import torch

classifier = pipeline("zero-shot-classification",
                      model="facebook/bart-large-mnli", device = torch.device("mps"))
candidate_labels = ['label1', 'label2', 'label3'] #list of labels to tag
classifier("SAMPLE TEXT INPUT" , candidate_labels)