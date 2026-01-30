import torch

@register_decomposition(aten.copy)
def copy(self, src, non_blocking=False):
    intermediate = src.to(self, non_blocking)
    # cheapest case
    if self.size() == intermediate.size() and self.stride() == intermediate.stride():
        return intermediate
    # next cheapest case
    if self.size() == intermediate.size() and self.is_contiguous():
        return intermediate.contiguous()
    # next cheapest case (expand_copy is guaranteed to return a contiguous tensor)
    if self.is_contiguous():
        return aten.expand_copy.default(intermediate, self.size())
    # expensive case
    # We need to return a tensor with the data of "src",
    # But with the size/stride/storage_offset of "self" (... and any other metadata! neg, conj, etc).
    # A problem for another day
    required_size = compute_required_storage_length(self.size(), self.stride(), self.storage_offset())
    out_buffer = torch.empty(required_size, dtype=self.dtype, device=self.device)
    out_buffer_updated = aten.as_strided_scatter(out_buffer, src, self.size(), self.stride(), self.storage_offset())
    return out_buffer_updated.as_strided(self.size(), self.stride(), self.storage_offset())