event_dim = transform.codomain.event_dim + max(domain_event_dim - base_event_dim, 0)

event_dim = transform.codomain.event_dim + max(reinterpreted_batch_ndims, 0)

py
transform_change_in_event_dim = transform.codomain.event_dim - transform.domain.event_dim
event_dim = max(
    transform.codomain.event_dim,  # the transform is coupled
    base_dist.event_dim + transform_change_in_event_dim  # the base dist is coupled
)

transform_change_in_event_dim = transform.codomain.event_dim - transform.domain.event_dim
event_dim = base_event_dim + transform_change_in_event_dim

event_dim = transform.codomain.event_dim + base_event_dim - transform.domain.event_dim