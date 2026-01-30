pt.tensor(5, dtype=pt.int64, device="mps").exp()  # crash
pt.tensor(5, dtype=pt.int64, device="mps").log()  # crash
pt.tensor(5, dtype=pt.int32, device="mps").exp()  # returns tensor(0., device='mps:0')
pt.tensor(5, dtype=pt.int32, device="mps").log()  # returns tensor(0., device='mps:0')