# Same bindings as 48875, but now implicitly grabs a private mempool
graph1.capture_begin()
graph1.capture_end()

# pool=... is new.  It hints that allocations during graph2's capture may share graph1's mempool
graph2.capture_begin(pool=graph1.pool())
graph2.capture_end()

# graph3 also implicitly creates its own mempool
graph3.capture_begin()
graph3.capture_end()