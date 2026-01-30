class Sampler:
	def __iter__(self):
		print('init list')
		return iter(list(range(10)))

for x in Sampler():
	print(x)

for i in sampler:
  print(i)