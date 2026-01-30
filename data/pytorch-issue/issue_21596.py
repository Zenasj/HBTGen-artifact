import torch as tt
dist = tt.distributions.Dirichlet(tt.ones(3))
support = dist.support
tform = tt.distributions.constraint_registry.biject_to(support)
dist_unconstrained = tt.distributions.TransformedDistribution(dist,tform.inv)

print(tform)
print(dist_unconstrained.sample())
print(dist_unconstrained.event_shape)

StickBreakingTransform()
tensor([ 0.3628, -1.9573])
torch.Size([3])

torch.Size([2])

py
class Transform(object):
    def transform_event_shape(self, event_shape):
        return event_shape

class StickBreakingTransform(Transform):
    def transform_event_shape(self, event_shape):
        return torch.Size((event_shape[0] - 1,))