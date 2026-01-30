def evaluate_neg_log_likelihood(mean, k, nb_r):
   import torch
   from torch.distributions.negative_binomial import NegativeBinomial
   mean = torch.tensor(mean)
   k = torch.tensor(k)
   nb_r = torch.tensor(nb_r)
   nb_p = nb_r / (nb_r + mean)
   nb = NegativeBinomial(total_count=nb_r, probs=nb_p)
   return f"nb_p = {nb_p};-log_p(pytorch) = {-nb.log_prob(k)};-log_p(mine) = {-(torch.lgamma(k + nb_r) - torch.lgamma(k + 1) - torch.lgamma(nb_r) + nb_r * torch.log(nb_r / (nb_r + mean)) + (k == 0.) * (mean == 0.) + k * torch.nan_to_num(torch.log(mean / (nb_r + mean)), nan=0.0, posinf=0.0, neginf=0.0))}, 'mode = {mode}'"

#    print("nb_p =", nb_p, ";-log_p(pytorch)=", -nb.log_prob(k), ";-log_p=", end="")
#    return -(torch.lgamma(k + nb_r) -
#             torch.lgamma(k + 1) -
#             torch.lgamma(nb_r) +
#             nb_r * torch.log(nb_r / (nb_r + mean)) +
#             (k == 0.) * (mean == 0.) + k * torch.nan_to_num(torch.log(mean / (nb_r + mean)), nan=0.0, posinf=0.0, neginf=0.0)
#             )


print("The negative log likelihood of k failures will happen before the nb_r th success given nb_p = nb_r / (nb_r + mean) for NB(nb_r, nb_p)")
print("evaluate_neg_log_likelihood(200, 400, 2)", evaluate_neg_log_likelihood(200, 400, 2))
print("evaluate_neg_log_likelihood(200, 200, 2)", evaluate_neg_log_likelihood(200, 200, 2))
print("evaluate_neg_log_likelihood(200, 100, 2)", evaluate_neg_log_likelihood(200, 100, 2))
print("evaluate_neg_log_likelihood(200, 0, 2)", evaluate_neg_log_likelihood(200, 0, 2))