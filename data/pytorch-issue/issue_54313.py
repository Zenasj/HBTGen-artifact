import torch


def run_model():
    model = torch.jit.load("./lst.pt", map_location=torch.device('cpu'))
    example = torch.ones(2, 11, 8)

    # print(example)

    model.eval()
    print(model(example))
    print(model.forward(example))


if __name__ == '__main__':
    run_model()