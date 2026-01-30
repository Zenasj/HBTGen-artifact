import torch
from torch.distributions import RelaxedBernoulli

torch.manual_seed(2)
dist = RelaxedBernoulli(temperature = torch.tensor(0.05), logits = torch.tensor(-5.0))
sample = dist.sample()
print('Sample: {:.8E}'.format(sample))
print('Log prob: {:4.2f}'.format(dist.log_prob(sample)))

class ClampedRelaxedBernoulli(RelaxedBernoulli):

    def sample(self, sample_shape=torch.Size()):
        """
        Generates a sample_shape shaped sample or sample_shape shaped batch of
        samples if the distribution parameters are batched. Samples first from
        base distribution and applies `transform()` for every transform in the
        list.
        """
        with torch.no_grad():
            x = self.base_dist.sample(sample_shape)
            for transform in self.transforms:
                x = transform(x)
            eps = torch.finfo(self.base_dist.logits.dtype).eps
            return x.clamp(min=eps, max=1 - eps)


    def rsample(self, sample_shape=torch.Size()):
        """
        Generates a sample_shape shaped reparameterized sample or sample_shape
        shaped batch of reparameterized samples if the distribution parameters
        are batched. Samples first from base distribution and applies
        `transform()` for every transform in the list.
        """
        x = self.base_dist.rsample(sample_shape)
        for transform in self.transforms:
            x = transform(x)
        eps = torch.finfo(self.base_dist.logits.dtype).eps
        return x.clamp(min=eps, max=1 - eps)