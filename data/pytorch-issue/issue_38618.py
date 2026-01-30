import torch

if __name__ == '__main__':
    device = 'cuda'

    kf = KalmanFilter(device)
    mean, covariance = kf.initiate(torch.tensor([10, 15, 0.5, 10], device=device))
    mean, covariance = kf.predict(mean, covariance)
    mean, covariance = kf.update(mean, covariance, torch.tensor([12, 20, 0.6, 11], device=device))

    mean_2, covariance_2 = kf.initiate(torch.tensor([12, 13, 0.7, 5], device=device))
    mean_2, covariance_2 = kf.predict(mean_2, covariance_2)
    mean_2, covariance_2 = kf.update(mean_2, covariance_2, torch.tensor([13, 14, 0.7, 8], device=device))

    squared_maha = kf.gating_distance(torch.cat((mean, mean_2), dim=0),
                                      torch.cat((covariance, covariance_2), dim=0),
                                      torch.tensor([[12, 20, 0.6, 11],
                                                    [20, 16, 0.4, 18]], device=device))
    print(squared_maha)