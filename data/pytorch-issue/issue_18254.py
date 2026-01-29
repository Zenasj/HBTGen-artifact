import torch
from torch import nn
from torch.distributions import RelaxedBernoulli

class ClampedRelaxedBernoulli(RelaxedBernoulli):
    def sample(self, sample_shape=torch.Size()):
        with torch.no_grad():
            x = self.base_dist.sample(sample_shape)
            for transform in self.transforms:
                x = transform(x)
            eps = torch.finfo(self.base_dist.logits.dtype).eps
            return x.clamp(min=eps, max=1 - eps)

    def rsample(self, sample_shape=torch.Size()):
        x = self.base_dist.rsample(sample_shape)
        for transform in self.transforms:
            x = transform(x)
        eps = torch.finfo(self.base_dist.logits.dtype).eps
        return x.clamp(min=eps, max=1 - eps)

class MyModel(nn.Module):
    def forward(self, input_tensor):
        temperature = input_tensor[0]
        logits = input_tensor[1]

        original_dist = RelaxedBernoulli(temperature=temperature, logits=logits)
        clamped_dist = ClampedRelaxedBernoulli(temperature=temperature, logits=logits)

        sample = original_dist.sample()  # Sample from original to potentially hit boundary

        orig_log_prob = original_dist.log_prob(sample)
        clamped_log_prob = clamped_dist.log_prob(sample)

        # Check if original had NaN and clamped is finite
        orig_nan = torch.isnan(orig_log_prob)
        clamped_finite = ~torch.isnan(clamped_log_prob)
        return orig_nan & clamped_finite  # Returns True where fixed

def my_model_function():
    return MyModel()

def GetInput():
    # Generate input with temperature and logit
    temperature = torch.rand(1) * 0.1  # Small temperature to replicate the issue
    logits = torch.randn(1) * 10  # Extreme logits to hit boundaries
    return torch.cat([temperature, logits])

# Okay, I need to create a complete Python code file based on the GitHub issue provided. Let me start by understanding the problem. The user is reporting an issue with PyTorch's RelaxedBernoulli distribution where samples at the boundaries (0 or 1) result in NaN log probabilities. The user provided a temporary fix by clamping the samples between eps and 1-eps in a subclass called ClampedRelaxedBernoulli.
# The goal is to generate a Python code file following the specified structure. The code must include a MyModel class that encapsulates the models from the issue. Since there's a comparison between the original and the fixed version (ClampedRelaxedBernoulli), I need to fuse them into a single MyModel. The model should have both versions as submodules and implement comparison logic.
# First, let me parse the input. The original code uses RelaxedBernoulli, and the fix is the ClampedRelaxedBernoulli. So the MyModel should have two distributions: the original and the fixed one. The model's forward method might take an input (like logits and temperature) and return some comparison between the two distributions' log_probs, checking for NaNs or differences.
# Wait, the user's reproduction code uses fixed parameters (temperature=0.05, logits=-5.0). The GetInput function should generate a random tensor that fits the input expected by MyModel. Let me think about the input shape. The original example uses a single scalar for logits (torch.tensor(-5.0)), but in general, the input to the distributions would be a tensor of logits and temperature. However, in the code structure provided, the input is generated via GetInput(). The original code's sample is a single value, so maybe the input here is the parameters for the distribution? Or perhaps the model expects the input to be the sample, but that might complicate things. Alternatively, the model could take the parameters (logits and temperature) and then compute both distributions and their log_probs for a generated sample.
# Alternatively, maybe the MyModel is supposed to take a sample and compute the log_prob from both distributions, then compare them. Hmm, but the user's example uses the same distribution's log_prob on its own sample. Wait, in the original code, the sample is from the distribution, so the log_prob should be valid, but in their case it's NaN. The fix ensures that the sample is clamped so that log_prob doesn't produce NaN.
# The MyModel needs to encapsulate both the original and fixed versions. So perhaps the MyModel will take a sample and compute log_probs from both distributions, then check if they are close or if there's a NaN.
# Wait, the user's fix is a subclass of RelaxedBernoulli, so maybe the MyModel should have two instances: one of the original RelaxedBernoulli and one of ClampedRelaxedBernoulli. Then, when given input parameters (logits and temperature), both distributions are initialized, a sample is generated from each, and their log_probs are compared?
# Alternatively, the MyModel could be structured to take the parameters (logits and temperature), generate a sample from the original and clamped distributions, then compute their log_probs and check if there's a difference.
# Alternatively, maybe the model's forward function takes the parameters and returns some comparison metric. Let me think of the structure.
# The user's code example has:
# dist = RelaxedBernoulli(temperature=torch.tensor(0.05), logits=torch.tensor(-5.0))
# So the parameters are temperature and logits. The input to the model would probably be these parameters. So in the code structure, the input tensor generated by GetInput() should be a tuple (temperature, logits), or perhaps a tensor with those values. Let me see the GetInput function.
# The user's code uses scalar values, but in a more general case, the input could be a batch of parameters. Let's assume that the input to MyModel is a tensor that contains the temperature and logits. Wait, but temperature and logits can be tensors of any shape compatible with each other. Alternatively, perhaps the input is a dictionary, but to make it simple, maybe the input is a tuple of two tensors. However, the GetInput function needs to return a single tensor or a tuple that can be passed to the model.
# Alternatively, perhaps the model expects the input to be the parameters, so in the code, the MyModel could take the temperature and logits as inputs, but in the code structure, the input is a single tensor. Hmm, this is a bit ambiguous. Let me think again.
# The user's example uses a single temperature (scalar) and a single logit (scalar). The GetInput function should return a tensor that can be used to initialize the distributions. So maybe the input is a tuple of two tensors (temperature, logits), but since the code structure requires GetInput to return a single tensor, perhaps we can combine them into a single tensor. Alternatively, perhaps the model's forward method takes the temperature and logits as separate parameters. Wait, but in the structure provided, the model is called as MyModel()(GetInput()), so the input from GetInput must be a single tensor that can be passed as the argument to the model's forward function.
# Hmm, this is a bit tricky. Let's see: the user's original code uses temperature and logits as parameters to the distribution. To make this into a model, perhaps the model takes these parameters as inputs. So in the model's forward, you might have something like:
# def forward(self, temperature, logits):
# But since the input from GetInput must be a single tensor, perhaps the input is a tensor that has both temperature and logits. For example, a tensor of shape (2,) where the first element is temperature and the second is logit. So GetInput would generate a tensor of shape (2,) with random values. Then, in the model's forward, split that into temperature and logits. That way, the input is a single tensor. Let's go with that approach.
# So the input shape would be (2,), where the first element is temperature (a scalar), the second is the logit(s). Wait, but the logit can be a tensor of any shape. Maybe the user's case is a single logit, but in general, the logit can be a batch. To simplify, let's assume that the input tensor has two elements: temperature (scalar) and a tensor of logits (could be a batch). Hmm, but how to structure that as a single tensor? Alternatively, perhaps the input is a dictionary, but the code structure requires a tensor. Alternatively, maybe the input is a tensor with two elements: temperature and logit. For example, a tensor of shape (2,). Then, in the model's forward, the first element is temperature, the second is the logit. But in the user's example, the logit is a scalar. So this could work.
# So the GetInput function would return a tensor of shape (2,), where the first element is a temperature (like 0.05) and the second is a logit (like -5.0). That way, the model can split them into temperature and logit tensors.
# Now, the model's structure: the MyModel class will have two distribution instances, the original and the clamped one. Wait, but distributions are not typically part of a nn.Module's parameters. Hmm. Alternatively, the model could take the temperature and logit as inputs, then create the distributions on the fly. Because in the forward method, when you have the temperature and logit, you can initialize the distributions each time.
# Wait, the user's fix is a subclass of RelaxedBernoulli, so perhaps the MyModel will use both the original and the clamped version. The forward method would take the temperature and logit, create instances of both distributions, generate a sample from each (or from one and use the other's log_prob?), then compare the log_probs.
# Alternatively, perhaps the model's forward method is designed to test the two distributions. Let me think of how the user's original test case works. The user samples from the original distribution and then computes the log_prob, which might be NaN. The clamped version would clamp the sample, so the log_prob is valid. So the model could, given parameters (temperature and logit), sample from the original distribution, compute the log_prob of that sample under both distributions, and check if they are close, or if there's a NaN.
# Alternatively, maybe the model is structured to return the log_prob from both distributions, so that their outputs can be compared. The model would return a tuple of the two log_probs. Then, in testing, one could see if the original has a NaN and the clamped version does not.
# But according to the problem statement, the goal is to fuse the models into a single MyModel, implementing the comparison logic from the issue (like using torch.allclose, etc.). The output should reflect their differences. So the MyModel's forward should return a boolean or some indicator of whether they differ.
# Putting this together, here's a possible structure:
# class MyModel(nn.Module):
#     def forward(self, input_tensor):
#         # Split the input into temperature and logits
#         temperature = input_tensor[0]
#         logits = input_tensor[1]
#         # Create original and clamped distributions
#         original_dist = RelaxedBernoulli(temperature=temperature, logits=logits)
#         clamped_dist = ClampedRelaxedBernoulli(temperature=temperature, logits=logits)
#         # Sample from original distribution (since clamped uses the same base)
#         sample = original_dist.sample()
#         # Compute log_probs for both distributions
#         orig_log_prob = original_dist.log_prob(sample)
#         clamped_log_prob = clamped_dist.log_prob(sample)
#         # Compare the two log_probs, checking for NaN in original and not in clamped
#         # Or check if they are close except where original is NaN
#         # The output could be a boolean indicating if there's a difference (like orig has NaN and clamped doesn't)
#         # Or return a tensor indicating differences
#         # The user's expected fix is to have clamped's log_prob not be NaN where original was
#         # So the comparison could be that clamped_log_prob is not NaN where original was
#         # So return a boolean indicating if the clamped fixed the NaN issue in this case
#         has_orig_nan = torch.isnan(orig_log_prob)
#         has_clamped_nan = torch.isnan(clamped_log_prob)
#         result = (has_orig_nan) & (~has_clamped_nan)
#         # Or return the difference between the two log_probs, or a boolean indicating if they differ in a meaningful way
#         return result
# But I need to structure this as a model that can be called with the input from GetInput(). The model's forward function must take the input (the tensor from GetInput) and return the comparison result.
# Wait, but the user's example uses the same distribution's sample for the log_prob. The original's log_prob might be NaN because the sample is exactly 0 or 1, leading to log(0) or log(1 - 1) which is log(0). So in the clamped version, the sample is clamped to avoid 0 or 1, so log_prob should be finite.
# Therefore, the model's forward could return a boolean indicating whether the clamped version's log_prob is not NaN where the original was. Alternatively, return both log_probs so that the difference can be checked.
# However, the problem requires the model to encapsulate both models as submodules. Wait, but the original and clamped distributions are not nn.Modules, they are from torch.distributions. Hmm, this complicates things because the model can't have them as submodules. Therefore, perhaps the model's forward function constructs them each time, using the input parameters. The model itself doesn't have parameters, so it's okay.
# Alternatively, maybe the ClampedRelaxedBernoulli is part of the model's structure, and the original is compared against it. Since the user provided the ClampedRelaxedBernoulli code, perhaps the model uses that as a submodule, and the original is part of the forward logic.
# Wait, the user's code defines ClampedRelaxedBernoulli as a subclass. So in the MyModel, we can have an instance of ClampedRelaxedBernoulli as a submodule, but the original RelaxedBernoulli isn't a submodule. Hmm, perhaps the model's forward function creates both distributions each time based on the input parameters.
# Therefore, the MyModel's forward function will:
# 1. Split the input into temperature and logits.
# 2. Create both distributions (original and clamped) using these parameters.
# 3. Generate a sample from the original distribution (since the clamped one's sample is clamped, but the original's sample might be at the boundary).
# 4. Compute the log_prob of this sample under both distributions.
# 5. Compare the two log_probs, perhaps checking if the clamped's log_prob is valid where the original was NaN, and return some boolean or tensor indicating the difference.
# The output of the model should be a result that indicates their difference, like a boolean tensor where True means the clamped version fixed the NaN.
# Now, the function my_model_function() should return an instance of MyModel. Since there are no parameters to initialize, it's straightforward.
# The GetInput() function needs to return a tensor that can be split into temperature and logits. The original example uses temperature=0.05 and logits=-5.0, but to make it general, GetInput can generate a tensor of shape (2,) with random values. However, to replicate the original bug scenario, maybe the input should sometimes have logits that push the sample to the boundary. Let's make the input a tensor of shape (2, 1) where the first element is temperature (e.g., small value) and the second is a logit (e.g., very negative or positive).
# Wait, the input should be a tensor that, when split, gives the temperature and logits. Let's structure GetInput to return a tensor of shape (2, ), where the first element is temperature (scalar), and the second is the logit (scalar). The input shape comment at the top should reflect this.
# Wait, the first line of the code should be a comment like:
# # torch.rand(B, 2, dtype=torch.float) ← since each sample has two elements (temperature and logit)
# Wait, but the user's example uses a single logit, but in general, the logit can be a tensor of any shape as long as it matches the batch dimensions. However, to keep it simple, perhaps the input is a 1D tensor of two elements. So the input shape would be (2, ), so the comment is:
# # torch.rand(B, 2, dtype=torch.float)
# Wait, but B is the batch size. If we want a batch of inputs, maybe the input is a 2D tensor where each row has temperature and logit. But for simplicity, perhaps the batch size is 1. So the GetInput function returns a tensor of shape (2, ) (no batch), but the code's first comment line needs to indicate the input shape. Let's think:
# The input is a tensor where the first element is the temperature (scalar) and the second is the logit (scalar). So the shape is (2, ), so the comment would be:
# # torch.rand(2, dtype=torch.float)
# Wait, but the user's example uses tensors with requires_grad? Not sure, but the dtype should be float.
# Alternatively, maybe the temperature and logit can be vectors. But given the original example uses scalars, perhaps keeping it simple is better.
# Therefore, the GetInput function will return a tensor of shape (2, ) with random values. For example:
# def GetInput():
#     temperature = torch.rand(1) * 0.1  # small temperature to replicate the issue
#     logits = torch.randn(1) * 10  # to get extreme values
#     return torch.cat([temperature, logits])
# Wait, but concatenating them into a single tensor of shape (2, ). That way, the input is a tensor of shape (2, ), and in the model's forward, we can split:
# temperature = input_tensor[0]
# logits = input_tensor[1]
# But in PyTorch, the distributions can handle tensors of any shape as long as they are compatible. Since the user's example uses scalars, this should work.
# Now, putting it all together:
# The MyModel's forward function would:
# def forward(self, input_tensor):
#     temperature = input_tensor[0]
#     logits = input_tensor[1]
#     original_dist = RelaxedBernoulli(temperature=temperature, logits=logits)
#     clamped_dist = ClampedRelaxedBernoulli(temperature=temperature, logits=logits)
#     # Sample from original distribution (since clamped's sample is clamped, but the original's could be at boundary)
#     sample = original_dist.sample()
#     # Compute log_probs
#     orig_log_prob = original_dist.log_prob(sample)
#     clamped_log_prob = clamped_dist.log_prob(sample)
#     # Compare: check if original had NaN and clamped doesn't
#     orig_nan = torch.isnan(orig_log_prob)
#     clamped_finite = ~torch.isnan(clamped_log_prob)
#     result = orig_nan & clamped_finite  # True where original was NaN and clamped is finite
#     return result
# Alternatively, return both log_probs as a tuple, but the requirement is to return a boolean or indicative output. The above returns a boolean tensor indicating where the clamped version fixed the NaN.
# Now, the ClampedRelaxedBernoulli is part of the code, so we need to include that class in the code. Since the user provided the code for ClampedRelaxedBernoulli, we can include it as is. However, the model's code must be in the MyModel class. Wait, but ClampedRelaxedBernoulli is a distribution, not a nn.Module. So the MyModel can use it in its forward.
# Wait, the code structure requires that the entire code is in a single Python code block, with the MyModel class and the other functions. So the ClampedRelaxedBernoulli class must be defined in the code.
# Wait, the user's code for ClampedRelaxedBernoulli is provided in the issue. Let me check:
# The user's code for the fix is:
# class ClampedRelaxedBernoulli(RelaxedBernoulli):
#     def sample(self, sample_shape=torch.Size()):
#         with torch.no_grad():
#             x = self.base_dist.sample(sample_shape)
#             for transform in self.transforms:
#                 x = transform(x)
#             eps = torch.finfo(self.base_dist.logits.dtype).eps
#             return x.clamp(min=eps, max=1 - eps)
#     def rsample(self, sample_shape=torch.Size()):
#         x = self.base_dist.rsample(sample_shape)
#         for transform in self.transforms:
#             x = transform(x)
#         eps = torch.finfo(self.base_dist.logits.dtype).eps
#         return x.clamp(min=eps, max=1 - eps)
# Wait, but the base_dist for RelaxedBernoulli is a Logistic distribution, perhaps. But in any case, the code is provided. So we can include that class in the code.
# Therefore, the code structure would be:
# - The ClampedRelaxedBernoulli class (copied from the issue's user's temporary fix)
# - The MyModel class, which uses both the original and clamped distributions
# - The my_model_function() which returns MyModel()
# - The GetInput() function generating the input tensor.
# Putting this all together in the required structure.
# Now, check the constraints:
# 1. Class name must be MyModel(nn.Module). Check.
# 2. The models (original and clamped) are compared. The MyModel encapsulates both as part of its forward logic. Since they are not submodules (as they are distributions), but the model's forward creates them each time, this should be acceptable.
# 3. GetInput() returns a tensor that works with MyModel. The input is a tensor of shape (2, ), which is split into temperature and logit. Check.
# 4. The code must not have test code or __main__ blocks. Check.
# 5. The entire code is in a single Python code block. Yes.
# Now, let's structure the code step by step.
# First, the ClampedRelaxedBernoulli class is needed. We can define it outside the MyModel, as it's a custom distribution.
# Then, the MyModel class's forward function uses it.
# Wait, but in the forward function, when creating clamped_dist, we need to pass the same parameters as the original. Since the parameters are temperature and logits, which are tensors.
# Now, code outline:
# Wait, but the input_tensor in the forward is a tensor of shape (2, ), so when splitting, temperature and logits are each 0-dimensional tensors (scalars). But when creating the distributions, the temperature and logits can be scalar tensors, which is acceptable.
# However, in PyTorch distributions, parameters can be tensors of any shape, as long as they are compatible. The sample() method would then produce a sample with the same batch shape as the parameters.
# In the user's example, the parameters are scalars, so the sample is a scalar. The log_prob would also be a scalar.
# The forward function's return is a boolean tensor indicating whether the clamped version fixed the NaN.
# Now, the first comment line must be:
# # torch.rand(B, 2, dtype=torch.float) ← since input has two elements (temperature and logit)
# Wait, the input is a tensor of shape (2, ), so the batch size B is 1? Or if the input can have a batch dimension, like B x 2, but in the code above, the GetInput returns a tensor of shape (2, ), so B is 1.
# The first line comment should indicate the input shape. Since the GetInput function returns a 1D tensor of length 2, the input shape is (2, ), so the comment would be:
# # torch.rand(2, dtype=torch.float)
# Wait, but the user might want to have a batch dimension. Let me see the problem again. The user's example uses a single sample, but the code should be general. The input tensor could be a batch of parameters. For example, if the input is of shape (B, 2), then each row is a pair of temperature and logit. The model's forward would process each batch element.
# In that case, the GetInput function should return a tensor of shape (B, 2), where B is a batch size. To make it more general, perhaps the input is a 2D tensor with batch size as first dimension. Let's adjust:
# The GetInput function can return a tensor with a batch dimension. Let's set B to 1 for simplicity, but the code should handle any B.
# Therefore, the first comment line would be:
# # torch.rand(B, 2, dtype=torch.float)
# The GetInput function can be written as:
# def GetInput():
#     B = 1  # Batch size
#     temperature = torch.rand(B) * 0.1  # Small temperature
#     logits = torch.randn(B) * 10  # Extreme logits
#     return torch.cat([temperature.unsqueeze(1), logits.unsqueeze(1)], dim=1)
# Wait, to make them into a (B, 2) tensor:
# Wait, temperature and logits are each B-length tensors. To combine them into a (B,2) tensor:
# input_tensor = torch.stack([temperature, logits], dim=1)
# Yes, that would make it (B,2). So:
# def GetInput():
#     B = 1  # Can be any batch size, but for simplicity, set to 1
#     temperature = torch.rand(B) * 0.1
#     logits = torch.randn(B) * 10
#     return torch.stack([temperature, logits], dim=1)
# So the input shape is (B, 2), hence the first comment line:
# # torch.rand(B, 2, dtype=torch.float)
# Now, in the forward function, the input_tensor is of shape (B, 2), so splitting:
# temperature = input_tensor[:, 0]
# logits = input_tensor[:, 1]
# Then, the distributions are initialized with these parameters (temperature and logits each of shape (B, )). The sample would then have shape (B, ), and the log_probs as well.
# The comparison would then be element-wise over the batch.
# The return value would be a boolean tensor of shape (B, ), indicating for each batch element whether the clamped version fixed the NaN in the original.
# This seems better. Let me adjust the code accordingly.
# Revised code:
# ```python
# import torch
# from torch import nn
# from torch.distributions import RelaxedBernoulli
# class ClampedRelaxedBernoulli(RelaxedBernoulli):
#     def sample(self, sample_shape=torch.Size()):
#         with torch.no_grad():
#             x = self.base_dist.sample(sample_shape)
#             for transform in self.transforms:
#                 x = transform(x)
#             eps = torch.finfo(self.base_dist.logits.dtype).eps
#             return x.clamp(min=eps, max=1 - eps)
#     def rsample(self, sample_shape=torch.Size()):
#         x = self.base_dist.rsample(sample_shape)
#         for transform in self.transforms:
#             x = transform(x)
#         eps = torch.finfo(self.base_dist.logits.dtype).eps
#         return x.clamp(min=eps, max=1 - eps)
# class MyModel(nn.Module):
#     def forward(self, input_tensor):
#         temperature = input_tensor[:, 0]
#         logits = input_tensor[:, 1]
#         original_dist = RelaxedBernoulli(temperature=temperature, logits=logits)
#         clamped_dist = ClampedRelaxedBernoulli(temperature=temperature, logits=logits)
#         sample = original_dist.sample()  # Sample from original to hit boundary
#         orig_log_prob = original_dist.log_prob(sample)
#         clamped_log_prob = clamped_dist.log_prob(sample)
#         # Check if original had NaN and clamped is finite
#         orig_nan = torch.isnan(orig_log_prob)
#         clamped_finite = ~torch.isnan(clamped_log_prob)
#         return orig_nan & clamped_finite  # Returns True where fixed
# def my_model_function():
#     return MyModel()
# def GetInput():
#     B = 1  # Batch size
#     temperature = torch.rand(B) * 0.1  # Small temperature to trigger boundary
#     logits = torch.randn(B) * 10  # Extreme logits to hit boundaries
#     return torch.stack([temperature, logits], dim=1)  # Shape (B, 2)
# ```
# Now, the input shape comment is correct as # torch.rand(B, 2, dtype=torch.float).
# Testing this code:
# When the input has a logit that's very negative (like -5 as in the example), the original sample might be 0.0, leading to log_prob NaN. The clamped_dist's sample would be slightly above 0, so log_prob is finite. The return value would be True for that batch element.
# This should satisfy the requirements. The MyModel encapsulates both distributions, compares their log_probs, and returns a boolean indicating if the clamped fixed the NaN.
# Also, the ClampedRelaxedBernoulli is correctly defined as a subclass of RelaxedBernoulli, with overridden sample and rsample methods to clamp the outputs.
# I think this meets all the constraints. The model is usable with torch.compile, as it's a nn.Module. The GetInput returns a valid input tensor. The code doesn't have any test blocks, just the required functions and classes.
# </think>
# ```python
# # torch.rand(B, 2, dtype=torch.float)  # Input shape: batch_size x [temperature, logit]
# import torch
# from torch import nn
# from torch.distributions import RelaxedBernoulli
# class ClampedRelaxedBernoulli(RelaxedBernoulli):
#     def sample(self, sample_shape=torch.Size()):
#         with torch.no_grad():
#             x = self.base_dist.sample(sample_shape)
#             for transform in self.transforms:
#                 x = transform(x)
#             eps = torch.finfo(self.base_dist.logits.dtype).eps
#             return x.clamp(min=eps, max=1 - eps)
#     def rsample(self, sample_shape=torch.Size()):
#         x = self.base_dist.rsample(sample_shape)
#         for transform in self.transforms:
#             x = transform(x)
#         eps = torch.finfo(self.base_dist.logits.dtype).eps
#         return x.clamp(min=eps, max=1 - eps)
# class MyModel(nn.Module):
#     def forward(self, input_tensor):
#         temperature = input_tensor[:, 0]
#         logits = input_tensor[:, 1]
#         original_dist = RelaxedBernoulli(temperature=temperature, logits=logits)
#         clamped_dist = ClampedRelaxedBernoulli(temperature=temperature, logits=logits)
#         sample = original_dist.sample()  # Sample from original distribution to test boundary cases
#         orig_log_prob = original_dist.log_prob(sample)
#         clamped_log_prob = clamped_dist.log_prob(sample)
#         # Compare if clamped version fixed NaN in original's log_prob
#         orig_nan = torch.isnan(orig_log_prob)
#         clamped_finite = ~torch.isnan(clamped_log_prob)
#         return orig_nan & clamped_finite  # Returns True where clamped fixed NaN
# def my_model_function():
#     return MyModel()
# def GetInput():
#     B = 1  # Batch size
#     temperature = torch.rand(B) * 0.1  # Small temperature to trigger boundary issues
#     logits = torch.randn(B) * 10  # Extreme logits to hit 0/1 boundaries
#     return torch.stack([temperature, logits], dim=1)  # Shape (B, 2)
# ```