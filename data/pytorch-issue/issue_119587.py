import torch
import torch.nn as nn

class M(torch.nn.Module):
    def forward(
        self, scores, score_thr, topk: torch.Tensor, results=None
    ):
        valid_mask = scores > score_thr
        scores = scores[valid_mask]
        valid_idxs = torch.nonzero(valid_mask).to(scores.device)
        # num_topk = min(topk, valid_idxs.size(0))
        num_topk = torch.minimum(topk, torch.tensor(valid_idxs.shape[0])).item()
        torch._constrain_as_size(num_topk)
        torch._check(scores.shape[0] > num_topk)
        # assert scores.shape[0] > num_topk
        # torch.sort is actually faster than .topk (at least on GPUs)
        scores, idxs = scores.sort(descending=True)
        # print(f"inside_scores: scores={scores}")
        # print(f"inside_scores: idxs={idxs}")
        scores = scores[:num_topk]
        # print(f"inside_scores. idxs[:num_topk]={idxs[:num_topk]}")
        topk_idxs = valid_idxs[idxs[:num_topk]]
        # print(f"after inside_scores: topk_idxs={topk_idxs}")
        keep_idxs, labels = topk_idxs.unbind(dim=1)
        # print(f"after inside_scores: scores={scores}")
        # print(f"after inside_scores: topk_idxs={topk_idxs}")
        # print(f"after keep_idxs: keep_idxs={keep_idxs}")
        # print(f"after labels: labels={labels}")
        filtered_results = None
        if results is not None:
            if isinstance(results, dict):
                filtered_results = {k: v[keep_idxs] for k, v in results.items()}
            elif isinstance(results, list):
                filtered_results = [result[keep_idxs] for result in results]
            elif isinstance(results, torch.Tensor):
                filtered_results = results[keep_idxs]
            else:
                raise NotImplementedError(
                    f"Only supports dict or list or Tensor, "
                    f"but get {type(results)}."
                )
        # print(f"after filtered_results: filtered_results={filtered_results}")
        return scores, labels, keep_idxs, filtered_results
score = torch.tensor(
    [[0.1, 0.3, 0.2], [0.12, 0.7, 0.9], [0.02, 0.8, 0.08], [0.4, 0.1, 0.08]]
)
bbox_pred = torch.tensor([[0.2, 0.3], [0.4, 0.7], [0.1, 0.1], [0.5, 0.1]])
score_thr = 0.15
nms_pre = torch.tensor(4)
inputs = (score, score_thr, nms_pre, dict(bbox_pred=bbox_pred))
ep = torch.export.export(M(), inputs)