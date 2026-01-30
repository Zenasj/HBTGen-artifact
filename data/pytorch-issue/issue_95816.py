import torch

m = Wishart(torch.eye(2), torch.Tensor([2]))
m.sample()  # Wishart distributed with mean=`df * I` and
            # variance(x_ij)=`df` for i != j and variance(x_ij)=`2 * df` for i == j

m = Wishart(torch.Tensor([2]), covariance_matrix=torch.eye(2))
m.sample()