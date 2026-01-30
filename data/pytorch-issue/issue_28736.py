for grad_list in [[grad_w, grad_b], [grad_w, None]]:
    for p, g in zip(l.parameters(), grad_list):
        p._grad = g.clone().view_as(p.data) if g is not None else g