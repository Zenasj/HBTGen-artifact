dp1 = dp.map.SequenceWrapper(range(10))
shuffle_dp1 = dp1.shuffle()
dp2 = dp.map.SequenceWrapper(range(10))
shuffle_dp2 = dp2.shuffle()
zip_dp = shuffle_dp1.zip(shuffle_dp2)
list(zip_dp)  # This used to fail