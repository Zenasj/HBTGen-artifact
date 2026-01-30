import torch

dist = torch.distributions.MultivariateNormal(torch.zeros(2), torch.eye(2))
x: torch.Tensor = dist.covariance_matrix  # ❌  E: reportAssignmentType
# Type "_lazy_property_and_property | Unknown" is not assignable to declared type "Tensor"