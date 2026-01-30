index = sub_block.body.indexing_from_args(
                                    (vars, reduction_vars)
                                )[
                                    _node.args[
                                        1 if _node.target == "index_expr" else 2
                                    ].args[0]
                                ]