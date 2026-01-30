import torch.nn as nn

import torch


def test_run_inference_twice(
    model: torch.nn.Module,
    x_dat: torch.Tensor,
) -> None:
    """Run the model in inference twice, updating states in-between."""
    # Gather the model's initial state, and generate new (random) ones.
    state_a = {key: val.clone() for key, val in model.state_dict().items()}
    state_b = {key: torch.rand_like(val) for key, val in state_a.items()}

    # Compute predictions in inference mode with the old states.
    model.eval()
    with torch.no_grad():
       preds_a = model(x_dat)
    model.train()

    # Change the model's state.
    model.load_state_dict(state_b)

    # Compute predictions in inference mode with the new states.
    model.eval()
    with torch.no_grad():
        preds_b = model(x_dat)
    model.train()

    # Assert that predictions defer (as they should).
    assert (preds_a != preds_b).any(), "Predictions are the same!"


def test_run_inference_twice_no_eval(
    model: torch.nn.Module,
    x_dat: torch.Tensor,
) -> None:
    """Copy of `test_run_inference_twice`, without `.eval()` / `.train()`."""
    # Gather the model's initial state, and generate new (random) ones.
    state_a = {key: val.clone() for key, val in model.state_dict().items()}
    state_b = {key: torch.rand_like(val) for key, val in state_a.items()}

    # Compute predictions in inference mode with the old states.
    with torch.no_grad():
       preds_a = model(x_dat)

    # Change the model's state.
    model.load_state_dict(state_b)

    # Compute predictions in inference mode with the new states.
    with torch.no_grad():
        preds_b = model(x_dat)

    # Assert that predictions defer (as they should).
    assert (preds_a != preds_b).any(), "Predictions are the same!"


# Linear model with sigmoid activation.
model = torch.nn.Sequential(
    torch.nn.Linear(128, 1),
    torch.nn.Sigmoid(),
)

# Stub input data.
x_dat = torch.randn(size=(4, 128))

# Run without torch.compile: both tests pass.
test_run_inference_twice(model, x_dat)
test_run_inference_twice_no_eval(model, x_dat)

# Run with torch.compile: the first test fails...
compiled_model = torch.compile(model)
test_run_inference_twice(compiled_model, x_dat)  # raises AssertionError
# ... but the second one works.
compiled_model = torch.compile(model)
test_run_inference_twice_no_eval(compiled_model, x_dat)

# Linear model with sigmoid activation.
model = torch.nn.Sequential(
    torch.nn.Linear(128, 1),
    torch.nn.Sigmoid(),
)
model = torch.compile(model)

# Stub input data.
x_dat = torch.randn(size=(4, 128))

# Gather the model's initial state, and generate new (random) ones.
state_a = {key: val.clone() for key, val in model.state_dict().items()}
state_b = {key: torch.rand_like(val) for key, val in state_a.items()}

# Turn the model to eval mode and run inference.
model.eval()
eval_a = model(x_dat)
# Update the model weights and run inference.
model.load_state_dict(state_b)
eval_b = model(x_dat)
# Turn the model to train mode and re-run inference.
model.train()
train_b = model(x_dat)

# These assertions pass.
assert (eval_a != eval_b).any()
assert (eval_b == train_b).all()

# Create the model anew and gather/generate states again.
model = torch.nn.Sequential(
    torch.nn.Linear(128, 1),
    torch.nn.Sigmoid(),
)
model = torch.compile(model)
state_a = {key: val.clone() for key, val in model.state_dict().items()}
state_b = {key: torch.rand_like(val) for key, val in state_a.items()}

# Do the same as before, but in eval mode, use `no_grad` context.
model.eval()
with torch.no_grad():
    eval_a = model(x_dat)
model.load_state_dict(state_b)
with torch.no_grad():
    eval_b = model(x_dat)
model.train()
train_b = model(x_dat)

# Both assertions fail!
assert (eval_a != eval_b).any()
assert (eval_b == train_b).all()

# Create the model anew and gather/generate states again.
model = torch.nn.Sequential(
    torch.nn.Linear(128, 1),
    torch.nn.Sigmoid(),
)
model = torch.compile(model)
state_a = {key: val.clone() for key, val in model.state_dict().items()}
state_b = {key: torch.rand_like(val) for key, val in state_a.items()}

# Run inference within `no_grad`, but in train mode.
with torch.no_grad():
    eval_a = model(x_dat)

model.load_state_dict(state_b)
with torch.no_grad():
    eval_b = model(x_dat)

# This passes: weights were properly used.
assert (eval_a != eval_b).any()

# This raises:
# RuntimeError: addmm(): functions with out=... arguments don't support
# automatic differentiation, but one of the arguments requires grad.
train_b = model(x_dat)

# Create the model anew and gather/generate states again.
model = torch.nn.Sequential(
    torch.nn.Linear(128, 1),
    torch.nn.Sigmoid(),
)
model = torch.compile(model)
state_a = {key: val.clone() for key, val in model.state_dict().items()}
state_b = {key: torch.rand_like(val) for key, val in state_a.items()}

# Run a forward pass in train mode.
train_a = model(x_dat)

# Switch to eval and run inference within `no_grad`.
model.eval()
with torch.no_grad():
    eval_a = model(x_dat)

# Update weights and run inference (still in eval + no_grad).
model.load_state_dict(state_b)
with torch.no_grad():
    eval_b = model(x_dat)

# Switch back to train and run the forward again.
model.train()
train_b = model(x_dat)

# These pass...
assert (train_a == eval_a).all()
assert (train_a != train_b).any()
# ... but this fails!
assert (eval_a != eval_b).any()

# Create the model anew and gather/generate states again.
model = torch.nn.Sequential(
    torch.nn.Linear(128, 1),
    torch.nn.Sigmoid(),
)
model = torch.compile(model)
state_a = {key: val.clone() for key, val in model.state_dict().items()}
state_b = {key: torch.rand_like(val) for key, val in state_a.items()}

# Run a forward pass in train mode, and one in eval mode *without no_grad*.
train_a = model(x_dat)
model.eval()
eval_a = model(x_dat)

# Update weights and run inference again, in eval + no-grad
model.load_state_dict(state_b)
with torch.no_grad():
    eval_b = model(x_dat)

# Switch back to train and run the forward again.
model.train()
train_b = model(x_dat)

# These all pass.
assert (train_a == eval_a).all()
assert (train_b == eval_b).all()
assert (train_a != train_b).any()
assert (eval_a != eval_b).any()
assert train_a.grad_fn is not None
assert train_b.grad_fn is not None
assert eval_a.grad_fn is not None
assert eval_b.grad_fn is None