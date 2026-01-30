s1.synchronize()
e_tok.record(s0) # Not sure if s1 or s2 but I guess it doesn't matter due to the s1.synchronize above
e_tok.synchronize()

self.assertTrue(s0.query())
self.assertTrue(s1.query())