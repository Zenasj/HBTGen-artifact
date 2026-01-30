for ra in ras:
                        log.debug("inserting runtime assert %s", ra.expr)
                        # Need to process ALL free symbols, not just unbacked ones
                        fvs = free_symbols(ra.expr)
                        missing = fvs - symbol_to_proxy.keys()
                        if missing:
                            i1 = sorted(missing)[0]
                            assert self.shape_env.is_unbacked_symint(i1), i1
                            ras_by_symbol.setdefault(i1, []).append(ra)