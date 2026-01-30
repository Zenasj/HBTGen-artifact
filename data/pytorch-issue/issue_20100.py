x = MultivariateNormal(t.ones(5, 3), t.eye(3))
x.log_prob(x.sample())
tensor([ -2.9737,  -4.3816,  -3.1320, -10.9442,  -3.1390])