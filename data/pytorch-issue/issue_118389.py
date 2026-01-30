diff
- self.assertLess(diff.max(), 15)
+ self.assertLess(diff.max(), 5)

out_value = sum([src[i + index_min] * w for i, w in zip(range(4), weights) ])