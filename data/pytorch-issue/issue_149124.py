import torch

class SimpleDataset(IterableDataset):
    def __init__(self, data, word2idx, window_size, num_neg_samples, neg_sampling_dist):
        self.data = data
        self.word2idx = word2idx
        self.window_size = window_size
        self.num_neg_samples = num_neg_samples
        # self.neg_sampling_dist = neg_sampling_dist.to("cuda")
        self.neg_sampling_dist = neg_sampling_dist

    def __iter__(self):
        for line in self.data:
            tokens = nltk.word_tokenize(line)
            token_ids = [
                self.word2idx.get(token, self.word2idx["<unk>"]) for token in tokens
            ]
            neg_sampling_dist = self.neg_sampling_dist.to("cuda")
            for i, center in enumerate(token_ids):
                start = max(0, i - self.window_size)
                end = min(len(token_ids), i + self.window_size + 1)
                for j in range(start, end):
                    if i != j:
                        context = token_ids[j]
                        negative_context = torch.multinomial(
                            neg_sampling_dist,
                            self.num_neg_samples,
                            replacement=True,
                        )
                        yield center, context, negative_context