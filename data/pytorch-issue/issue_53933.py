import torch

@torch.jit.script
def _prepare_decoder_input_ids_for_generation(
    input_ids: torch.LongTensor, decoder_start_token_id: int = None, bos_token_id: int = None
) -> torch.LongTensor:
    decoder_start_token_id = 47  # for simplicity
    decoder_input_ids = (
        torch.ones((input_ids.shape[0], 1), dtype=input_ids.dtype, device=input_ids.device)
        * decoder_start_token_id
    )
    return decoder_input_ids