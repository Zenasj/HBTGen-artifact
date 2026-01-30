def test_barrier_timeout_rank_tracing(self):
        N = 3 

        store = dist.HashStore()

        def run_barrier_for_rank(i: int):
            if i != 0:
                import time;time.sleep(1)  # Let some thread sleep for a while
            try:
                store_util.barrier(
                    store,
                    N,
                    key_prefix="test/store",
                    barrier_timeout=0.1,
                    rank=i,
                    rank_tracing_decoder=lambda x: f"Rank {x} host",
                    trace_timeout=0.01,
                )
            except Exception as e:
                return str(e)
            return ""