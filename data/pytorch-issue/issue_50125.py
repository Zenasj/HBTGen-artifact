import torch

@torch.jit.script
def _viterbi_decode_onnx(emissions: torch.Tensor,
                    mask: torch.Tensor,
                    start_transitions: torch.Tensor,
                    end_transitions: torch.Tensor,
                    transitions: torch.Tensor) -> List[torch.Tensor]:

    score = start_transitions + emissions[0]
    history: List[torch.Tensor] = [] 

    # Viterbi algorithm recursive case: we compute the score of the best tag sequence
    # for every possible next tag
    
    for i in range(1, 70):
        
        broadcast_score = score.unsqueeze(2)
        broadcast_emission = emissions[i].unsqueeze(1)
      
        next_score = broadcast_score + transitions + broadcast_emission
        next_score, indices = next_score.max(dim=1)
        
        score = torch.where(mask[i].unsqueeze(1), next_score, score) #this is where it fails. It's the where op, not indexing/squeezing
        history.append(indices)