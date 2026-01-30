import torch.nn as nn

import torch

class SimpleModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = torch.nn.Linear(5, 10)

    def forward(self, x):
        return self.fc1(x)


def try_load_state_dict(model: SimpleModel, state_dict: dict[str, torch.Tensor], strict: bool):
    print(f"state_dict.keys(): {state_dict.keys()}")
    print(f"strict={strict}")

    print(f"Calling load_state_dict()...")
    try:
        load_result = model.load_state_dict(state_dict, strict=strict)
        print(f"load_result.missing_keys: {load_result.missing_keys}")
        print(f"load_result.unexpected_keys: {load_result.unexpected_keys}")
    except Exception as e:
        print(f"Raised exception: {e}")

def main():
    simple_model = SimpleModel()

    print("Initialized SimpleModel.")
    print(f"simple_model.state_dict().keys(): {simple_model.state_dict().keys()}")
    
    print("\nCase 1: Correct state_dict, strict=True")
    print("-----")
    state_dict = simple_model.state_dict()
    try_load_state_dict(simple_model, state_dict, strict=True)

    print("\nCase 2: state_dict with 'fc1.bad_key', strict=False")
    print("-----")
    state_dict = simple_model.state_dict()
    state_dict["fc1.bad_key"] = torch.randn(5, 10)
    try_load_state_dict(simple_model, state_dict, strict=False)

    print("\nCase 3: state_dict with 'fc1.bad_key', strict=True")
    print("-----")
    state_dict = simple_model.state_dict()
    state_dict["fc1.bad_key"] = torch.randn(5, 10)
    try_load_state_dict(simple_model, state_dict, strict=True)

    print("\nCase 4: state_dict with 'fc1.weight.bad_suffix', strict=False")
    print("-----")
    state_dict = simple_model.state_dict()
    state_dict["fc1.weight.bad_suffix"] = torch.randn(5, 10)
    try_load_state_dict(simple_model, state_dict, strict=False)

    print("\nCase 5: state_dict with 'fc1.weight.bad_suffix', strict=True")
    print("-----")
    state_dict = simple_model.state_dict()
    state_dict["fc1.weight.bad_suffix"] = torch.randn(5, 10)
    try_load_state_dict(simple_model, state_dict, strict=True)


if __name__ == "__main__":
    main()