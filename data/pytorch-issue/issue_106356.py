# calling .debug_str() on a FusedSchedulerNode
buf0_buf1: FusedSchedulerNode(NoneType)
buf0_buf1.writes = [MemoryDep('buf0', c0, {c0: 10}), MemoryDep('buf1', c0, {c0: 10})]
buf0_buf1.unmet_dependencies = []
buf0_buf1.met_dependencies = [MemoryDep('arg0_1', c0, {c0: 100}), MemoryDep('arg1_1', c0, {c0: 10})]
buf0_buf1.users = None
buf0_buf1.snodes = ['buf0', 'buf1']