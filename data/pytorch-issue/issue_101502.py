import torch

feat_embedding_packed = pack_padded_sequence(feat_embedding_batched, seq_lens_batched, batch_first=True, enforce_sorted=False)

# Dummy encodings for goal nodes
dummy_enc = torch.zeros_like(node_encodings)
node_encodings = torch.cat((node_encodings, dummy_enc), dim=1)
# Gather node encodings along traversed paths
node_enc_selected = node_encodings[batch_idcs, traversals_batched]