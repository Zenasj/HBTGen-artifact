{{ modification(
    subgraph_number=0,
    output_name="post_mod_scores",
    score="qk",
    out="qk"
) | indent_except_first(1) }}

{{load_input("B", "b", ("idx_m", "idx_n"), mask=None if EVEN_K else "b_mask", indent_width=8)}}