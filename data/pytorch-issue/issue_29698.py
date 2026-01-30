def _kl_transformed_transformed(p, q):
    if p.transforms != q.transforms:
        raise NotImplementedError
    if p.event_shape != q.event_shape:
        raise NotImplementedError
    # extra_event_dim = len(p.event_shape) - len(p.base_dist.event_shape)
    extra_event_dim = len(p.event_shape)
    base_kl_divergence = kl_divergence(p.base_dist, q.base_dist) #call to indep_indep below
   #this will again sum over kl_divergence for each entry in batch
    return _sum_rightmost(base_kl_divergence, extra_event_dim)

@register_kl(Independent, Independent)
def _kl_independent_independent(p, q):
    shared_ndims = min(p.reinterpreted_batch_ndims, q.reinterpreted_batch_ndims)
    p_ndims = p.reinterpreted_batch_ndims - shared_ndims
    q_ndims = q.reinterpreted_batch_ndims - shared_ndims
    p = Independent(p.base_dist, p_ndims) if p_ndims else p.base_dist
    q = Independent(q.base_dist, q_ndims) if q_ndims else q.base_dist
    kl = kl_divergence(p, q)
    if shared_ndims:
       #this line gets called when base_dist is Gaussian
        kl = sum_rightmost(kl, shared_ndims)
    return kl