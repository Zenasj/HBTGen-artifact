import torch
def test_erfinv():
    for device in ['cpu', 'mps']:
        x = torch.tensor([0.1, 0.2, 0.3, 0.4, 0.5], device=device)
        y = x[2:].erfinv()

        x2 = torch.tensor([0.3, 0.4, 0.5], device=device)
        y2 = x2.erfinv()

        print(y)
        print(y2)

        torch.testing.assert_close(y, y2)
        print(f"{device} passes.")

test_erfinv()