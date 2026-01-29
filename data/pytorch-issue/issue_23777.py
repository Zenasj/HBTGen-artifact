# torch.rand(B, L, dtype=torch.long)
import torch
import torch.nn as nn

class MyModel(nn.Module):
    def __init__(self, vectors):
        super(MyModel, self).__init__()
        self.emb = nn.Embedding.from_pretrained(vectors, freeze=False, padding_idx=0)
        self.emb2 = nn.Embedding.from_pretrained(vectors, freeze=False, padding_idx=0)
        self.linear = nn.Linear(128, 2)
        
    def forward(self, s):
        s = self.emb(s)
        s = self.linear(s)
        s = s.sum(dim=1)
        return s

def my_model_function():
    vectors = torch.randn(1000, 128)
    return MyModel(vectors)

def GetInput():
    return torch.randint(1, 1000, (100, 30), dtype=torch.long)

# Okay, let's tackle this problem. The user wants me to generate a complete Python code file based on the GitHub issue provided. The issue is about inconsistent model behavior when loading the state dict in different ways. The key points from the issue are:
# First, the problem occurs when there are two embedding layers. The user noticed that when they use two embeddings, the model's outputs differ depending on whether they load the state dict before or after moving the model to the GPU. However, if they remove one embedding layer, the problem goes away. The comments in the issue explain that the issue arises because both embeddings share the same 'vectors' tensor and are not frozen. Since freeze is set to False, the embeddings are trainable, and modifying one affects the other because they reference the same data.
# The task is to create a single Python code file that encapsulates the problem described. The structure must include MyModel as a class, a function to create the model, and GetInput to generate the input tensor. Also, since the issue involves comparing two models (or methods), I need to fuse them into a single MyModel that includes both models as submodules and implements the comparison logic.
# First, I need to structure MyModel. The original Model has two embeddings and a linear layer. The problem comes from the two embeddings sharing the same vectors. So, in MyModel, perhaps we need to have both embeddings but ensure they are using the same vectors. Wait, but the user's code in the issue shows that both emb and emb2 are initialized from the same vectors, and since freeze is False, their weights are being updated during training. However, because they share the same underlying tensor, modifying one's weights affects the other. That's the root cause.
# So, in the MyModel class, I need to replicate the original Model's structure. The model should have two embeddings initialized from the same vectors. The forward pass uses only the first embedding, but the second exists and is not used, but its weights are still part of the model's parameters.
# Next, the functions. The my_model_function should return an instance of MyModel, initialized with the vectors. The GetInput function needs to generate a random tensor of the correct shape. Looking at the original code, the input is a tensor of shape (100,30), integers between 1 and 999, since the embeddings have 1000 entries (vectors is 1000x128). So the input shape is (B, L) where B is batch size and L is sequence length. The original code uses 100 samples of 30 elements each. The GetInput function should generate a random tensor of that shape, with the correct dtype (long, since embeddings take long indices) and device (probably CPU, but since the model can be moved to GPU, the input should be compatible).
# The user's code had an assert that the outputs of different loading methods were the same, but they failed. To encapsulate this into MyModel, perhaps the model needs to compare the two methods internally. Wait, the problem is about how the model is loaded and the order of moving to GPU and loading. The user's code has three inference methods (m1, m2, m3) which are different loading orders. The MyModel should encapsulate the comparison between these methods? Hmm, according to the special requirements, if multiple models are discussed together, they should be fused into a single MyModel with submodules and comparison logic.
# Wait the user's original code has three different inference methods (m1, m2, m3). The problem is that the outputs differ between these methods. The MyModel needs to include both models (like m1 and m2) as submodules, and in its forward, maybe run both and compare? Or perhaps the MyModel is the original model, but the code should include the logic to test the different loading paths?
# Wait the task says: "If the issue describes multiple models (e.g., ModelA, ModelB), but they are being compared or discussed together, you must fuse them into a single MyModel". Here, the issue is not comparing two different models, but two different loading procedures. However, the two different loading methods (m1 and m2) are instances of the same Model class but loaded differently. So perhaps the MyModel should have both models as submodules, and the forward would run both and compare their outputs?
# Alternatively, maybe the MyModel is the original model, and the functions will handle the different loading paths. But the problem is that the user's code's issue is that the different loading orders produce different outputs. To encapsulate this into the model, perhaps the MyModel's forward function will return the outputs of both methods (m1 and m2) so that their difference can be checked? Or perhaps the model's forward is designed to test this scenario.
# Alternatively, the problem is that the two embeddings share the same vectors, leading to unexpected behavior when training. The MyModel should be the original Model, with two embeddings from the same vectors and freeze=False. The GetInput function provides the input, and the model's forward is as before. But the special requirement says that if multiple models are discussed, they should be fused. Since the problem arises from the two embeddings, perhaps the model structure is correct, and the MyModel is the original Model. The functions my_model_function would return an instance of that model. The GetInput function is straightforward.
# Wait, the user's code in the issue is the minimal reproduction. The problem is that when you have two embeddings initialized from the same vectors (with freeze=False), the order of loading and moving to GPU affects the output. The key is that the vectors are not copied, so both embeddings reference the same underlying tensor. Therefore, when you train the model, the gradients from both embeddings affect the same tensor. But since in the forward pass only the first embedding is used, the second is never used, but its weights are still part of the model parameters. So during training, the optimizer will update both embeddings' weights, but since they share the same tensor, they are the same. Wait no, when you call from_pretrained, the weights are copied if freeze is False? Wait, the documentation says that from_pretrained copies the tensor if freeze is False. Wait, let me check: the Embedding.from_pretrained documentation says that if freeze is False, the tensor is copied and becomes a parameter. Wait, actually, when freeze is False, the tensor is copied into a parameter, so each embedding would have their own copy. Wait, but in the user's code, they are both initialized from the same vectors variable. So vectors is a tensor, and when you create the first embedding, it's copied into a parameter (since freeze=False). Then the second embedding is also initialized from the same vectors variable. But since freeze=False again, it would copy the vectors again. But if vectors was modified after the first initialization, then the second would have the original vectors. Wait but in the user's code, the vectors are generated once before creating the model. So when creating the two embeddings, they both start with the same initial vectors. Then during training, the gradients from the first embedding (since it's used in the forward) will update its weights, but the second embedding is not used, so its gradients are zero. Wait, but the second embedding is part of the model's parameters, so when you do optimizer.step(), the gradients for the second embedding's weights are zero. So their weights would stay the same as the initial vectors. Wait, but the first embedding's weights are updated. So the two embeddings would have different weights after training? But the user's problem is that when loading the state dict in different orders, the outputs differ.
# Hmm, maybe the problem is related to the device where the model is loaded. When you move the model to GPU before or after loading the state dict, the tensors' devices may affect how the parameters are stored. The issue's comments explain that the problem is due to the two embeddings sharing the same vectors (but I thought they are copies). Wait the comment from rgommers says that the root cause is that both embeddings use the same vectors input with freeze=False. The documentation says that when freeze is False, the tensor is copied into a parameter. Therefore, if you create two embeddings from the same vectors, each will have their own copy. However, in the user's code, the vectors variable is a tensor that is passed to both embeddings. So when the first embedding is created, it makes a copy (since freeze=False). The second embedding also makes a copy of the original vectors. But if vectors is a tensor that is modified after the first embedding is created, then the second would have the original. However, in the user's code, vectors is created once and then passed to both embeddings. So each embedding has its own copy of the original vectors. Therefore, during training, the first embedding's weights are updated, but the second remains as the original. But since the second is not used in the forward, its gradients are zero. So why would this cause the problem?
# Ah, the issue's comment says that the problem arises because the vectors are modified in place. Wait, if freeze=False, the parameter is a copy, so modifying the vectors after creating the embeddings would not affect the embeddings. Wait, perhaps the user made a mistake in the code. Let me re-examine the user's code.
# Looking at the user's code:
# vectors = torch.randn(1000, 128)
# Then, in the model's __init__, they do:
# self.emb = nn.Embedding.from_pretrained(vectors, freeze=False, ...)
# self.emb2 = nn.Embedding.from_pretrained(vectors, freeze=False, ...)
# So when creating emb and emb2, each is initialized with vectors. Since freeze is False, each creates a copy of vectors as their weight parameter. Therefore, the two embeddings start with the same initial weights. However, during training, the gradients for emb's weights will be computed (since it's used in forward), but the gradients for emb2's weights will be zero (since it's not used). So the optimizer.step() would update emb's weights, but leave emb2's weights as the initial vectors. Therefore, the two embeddings have different weights after training.
# Wait, but in the forward pass, only the first embedding is used. So the second embedding's weights are never used, so their gradients are zero. Therefore, when you save the model's state_dict, both embeddings' weights are saved. When you load the state_dict into a new model, the new model's embeddings will have the updated weights for emb and the original (or updated?) weights for emb2?
# Wait no. The problem arises when loading the state_dict in different orders. Let me think about the different loading methods.
# Case 1: model is initialized on CPU, then moved to GPU, then load state dict. The state_dict's tensors are on CPU, so when loaded into the model on GPU, they are moved to GPU.
# Case 2: model is initialized on CPU, load the state_dict (which is on CPU), then moved to GPU. The state_dict's weights are loaded on CPU, then the model is moved to GPU, so the tensors are moved.
# Wait, but the issue's problem is that the outputs are different between the two cases. The comment says that the problem is because the two embeddings share the same vectors. Wait, the comment from rgommers says:
# "The root cause is that both embeddings use the same vectors input with freeze=False: The default for freeze is True; using that all asserts pass. Changing the use of the second embedding to use a unique tensor input: also makes the asserts pass. tl;dr the input vector gets modified in place, and since that's done in two separate embeddings, the results become inconsistent."
# Wait, that's confusing. If freeze is False, then the embeddings have their own copies of vectors, so modifying vectors after creating the embeddings would not affect the embeddings. But if freeze is True, then the embeddings' weights are tied to the original vectors, so modifying vectors would change the embeddings. But in the user's code, freeze is set to False. So according to that, the two embeddings have their own copies, so modifying vectors after creation shouldn't affect them. However, the comment says that the problem is because the input vectors are modified in place. Hmm, perhaps there's a misunderstanding here.
# Alternatively, maybe the user is using the same vectors tensor for both embeddings, and if vectors is a parameter that's being updated, but that's not the case. Wait, in the code, vectors is a tensor created with torch.randn and passed to the embeddings. Since freeze is False, each embedding makes a copy. So the vectors variable itself is not part of the model's parameters. Therefore, the embeddings have their own parameters (copied from vectors) which are trainable. So during training, only the first embedding's parameters are updated (since it's used in the forward), while the second's parameters remain as initial (since their gradients are zero).
# Therefore, the problem must be something else. The comment says that the issue is resolved by making the second embedding use a deep copy of vectors. That would ensure that the second embedding's initial weights are not affected by any changes to the first's. Wait but if freeze is False, then the initial vectors are copied when creating the embedding. So why would they need to deep copy vectors? Unless the vectors tensor was being modified between the two embeddings' creation. But in the user's code, vectors is created once, and both embeddings are created immediately after, so the initial vectors are the same for both. 
# Hmm, perhaps the issue is related to the device on which the model is loaded. For example, when you load the state dict onto a model that's on CPU vs GPU. But the problem description mentions that when using one embedding (removing emb2), the problem goes away. That suggests that the presence of the second embedding is causing some unexpected behavior during the loading and device transfer.
# Alternatively, the problem is that when you load the state dict into the model in different orders (before or after moving to GPU), the device of the parameters might not be properly set, leading to different computations. For instance, if the model is moved to GPU first, then the loaded parameters are on GPU, but if loaded first on CPU then moved, the parameters are moved. However, in PyTorch, loading a state dict should handle the device automatically. But maybe there's a bug in the way the state dict is saved or loaded when there are multiple parameters with the same underlying data (though they shouldn't be).
# Wait the user's problem is that the asserts fail. The first two asserts (o1 vs o2, o1 vs o3) fail. The third (o2 vs o3) passes. So when loading the model in different ways, the outputs differ. The comment explains that this is because the two embeddings share the same vectors, leading to modifications in place. But how does that cause the output differences based on load order?
# Wait maybe the issue is that when you create the model on CPU first and then move to GPU, the parameters are moved to GPU. However, the state dict is saved from a model that was on GPU. So when you load the state dict into a CPU model, the tensors in the state dict are on GPU, so you have to move them to CPU first. But the user's code may have an error in loading. Wait looking at the user's code:
# In the saving part: torch.save(model.state_dict(), 'tmp.pt'), where model is on CUDA. So the saved state_dict contains tensors on CUDA. 
# Then, when loading into m1: m1 is created on CPU, then moved to CUDA. The load_state_dict is called with torch.load('tmp.pt'), which loads the tensors on CUDA. But m1's parameters are on CPU. So when loading, the tensors in the state dict are on CUDA, but the model's parameters are on CPU. This would cause an error unless map_location is used. Wait, in the user's code, when they load into m1:
# m1 = Model(vectors).to('cuda') → model is on GPU.
# Then m1.load_state_dict(torch.load('tmp.pt')). The saved state_dict's tensors are on CUDA. The model's parameters are on CUDA, so it should match. 
# Wait, the saved model was on CUDA, so the state_dict tensors are on CUDA. So when you load into a model on CUDA, that's okay. 
# For m2: m2 is created on CPU (Model(vectors)), then load_state_dict(torch.load(...)) → the state_dict's tensors are on CUDA, so the model's parameters (on CPU) would need to have the tensors moved to CPU. But when you then do m2.to('cuda'), that moves the parameters to CUDA. 
# Wait, but the load_state_dict would require that the tensors in the state_dict match the device of the model's parameters. If the model is on CPU, and the state_dict tensors are on CUDA, then the load would fail unless you specify map_location='cpu' when loading. 
# Ah! This is probably the issue. The user's code didn't use map_location when loading the state_dict. So when they do:
# m2 = Model(vectors)
# m2.load_state_dict(torch.load('tmp.pt')) → the saved state_dict has tensors on CUDA, and m2 is on CPU. So this would raise an error unless the tensors are moved to CPU. 
# Wait, but in the user's code, perhaps they are using PyTorch 1.1.0, which might have different behavior. Alternatively, maybe the user's code has a bug here, which is causing the problem. But according to the comments, the problem is resolved when using freeze=True, which suggests that the issue is not about device mismatch but about the embeddings' shared parameters.
# Wait the comment from rgommers says that changing the second embedding to use a deep copy of vectors fixes the problem. That implies that the two embeddings' initial weights were sharing the same tensor. But with freeze=False, they should have their own copies. So why would they share? 
# Wait, perhaps the user made a mistake in creating the vectors. Let me think again. The code:
# vectors = torch.randn(1000, 128)
# Then, in the model's __init__, both embeddings are created from vectors. Since freeze=False, each creates a copy. So the two embeddings have their own copies. Therefore, modifying one's weights shouldn't affect the other. 
# But during training, the first embedding's gradients are computed and updated, while the second's are not. So their weights diverge. When the model is saved, both embeddings' weights are saved. 
# When you load the model in different orders, perhaps the order affects how the parameters are loaded, but that's unlikely. Alternatively, the problem is that the two embeddings' weights are being updated in a way that's dependent on the device. For example, when moving to GPU, the parameters are stored in different ways, leading to different computations.
# Alternatively, the user's problem arises from the fact that when you load the state_dict into a model on CPU (m2 before to('cuda')), the parameters are on CPU, but the saved state_dict's tensors are on CUDA. So the load_state_dict would require that the tensors are moved to CPU. But if the user didn't use map_location='cpu', then it would fail. However, in the code provided, the user didn't include error handling, so maybe the code actually has an error. But according to the issue, the asserts are failing, not throwing an error. 
# Wait the user's code in the issue has the following line for loading m2's state_dict:
# m2.load_state_dict(torch.load('tmp.pt'))
# But the saved state_dict is from a model on CUDA, so the tensors are on CUDA. The model m2 is on CPU. Thus, the load would fail because the tensors in the state_dict are on CUDA but the model's parameters are on CPU. To fix this, you need to use map_location='cpu' when loading. 
# Wait, but in PyTorch, when you load a state_dict from a different device, you can load it into a model on CPU by specifying map_location='cpu' in torch.load. Otherwise, it will try to load the tensors to the same device they were saved on. So in the user's code, when loading m2 (which is on CPU), the load would fail unless they use map_location. 
# Ah! This is a crucial point. The user's code may have a bug here. Because when they load the state_dict into m2 (which is on CPU), the tensors are on CUDA (from the saved model), so they can't be assigned to CPU parameters. This would raise an error, but the user's code's asserts are failing, implying that the code runs. 
# Wait maybe in PyTorch 1.1.0, the behavior was different. Let me check the documentation. The torch.load function has a map_location parameter. If not specified, it loads the tensors to the same device they were saved. So if the model was saved on CUDA, then loading it on a CPU model would require map_location='cpu'. 
# Therefore, in the user's code, when they load into m2 (CPU model), they should have done:
# m2.load_state_dict(torch.load('tmp.pt', map_location='cpu'))
# Otherwise, it would throw an error. Since the user's code didn't include that, perhaps that's the real issue. But according to the comments, the problem is about the embeddings sharing vectors. 
# Hmm, the user's issue description says that the problem disappears when using one embedding. So the presence of the second embedding causes the problem. The comment from rgommers says that the problem is because the two embeddings share the same vectors, leading to in-place modifications. 
# Putting this together, perhaps the key is that when freeze is False, the embeddings have their own parameters, but during training, the gradients from the first embedding are computed and applied, while the second's gradients are zero. However, the second's parameters are still part of the model's parameters, so when the model is saved, all parameters are saved. When loading the model in different ways (before or after moving to GPU), the parameters are loaded correctly, but the forward pass uses only the first embedding. However, due to the presence of the second embedding, which has different weights, maybe there's some interaction, or perhaps the order of parameter registration affects the computation.
# Alternatively, the problem is that when you load the state_dict into a model on CPU and then move it to GPU, the parameters are moved, but perhaps some operations are done in a way that depends on the device order. 
# Alternatively, the user's code has a bug in the way they save the model. They wrote torch.save(model.state_dict, ...) but should have used model.state_dict(). The issue's code shows:
# torch.save(model.state_dict, 'tmp.pt')
# Wait, that's a mistake! The user is saving the state_dict method itself, not the result of calling it. The correct way is model.state_dict(). So the saved file contains the method, not the actual state dict. That would definitely cause errors when loading. 
# Wait, looking at the user's code in the issue's "To Reproduce" section:
# They have:
# torch.save(model.state_dict, 'tmp.pt')
# But the correct way is torch.save(model.state_dict(), 'tmp.pt'). So that's a critical error. If the user saved the method instead of the state dict, then loading would fail. 
# Ah! This is a possible explanation. If the user saved the method (state_dict), then when they load it, it would not be a valid state_dict, leading to errors. But the user's code's asserts are failing, not throwing an error. 
# But the user's comments indicate that the problem is resolved by changing the second embedding's initialization to use a deep copy. That suggests that the error is not due to this mistake. 
# Hmm, this is getting complicated. Let's proceed to structure the code as per the problem's requirements.
# The user's original code has a Model class with two embeddings. The problem is that when you have two embeddings initialized from the same vectors with freeze=False, the order of loading and moving to device causes different outputs. The MyModel should encapsulate this scenario. 
# Following the requirements:
# The MyModel must be a class derived from nn.Module. The code must have:
# - A comment at the top indicating the input shape. The input is a tensor of integers (indices) with shape (batch_size, sequence_length). In the example, it's (100,30). So the input shape comment should be torch.rand(B, L, dtype=torch.long). 
# The class MyModel should have the same structure as the original Model. So:
# class MyModel(nn.Module):
#     def __init__(self, vectors):
#         super().__init__()
#         self.emb = nn.Embedding.from_pretrained(vectors, freeze=False, padding_idx=0)
#         self.emb2 = nn.Embedding.from_pretrained(vectors, freeze=False, padding_idx=0)
#         self.linear = nn.Linear(128, 2)
#     
#     def forward(self, s):
#         s = self.emb(s)
#         s = self.linear(s)
#         s = s.sum(dim=1)
#         return s
# Wait but in the original code, the vectors are passed to the model's __init__. So the my_model_function must return an instance initialized with vectors. But how are vectors generated? The GetInput function must return a tensor compatible with the model's input. 
# The my_model_function should return an instance of MyModel, but needs to have the vectors. Since the vectors are generated as random in the original code, perhaps in the generated code, we can create them inside the function. Or perhaps the function requires vectors as an argument. Wait, the original code's model is initialized with vectors as an argument. So the my_model_function must take vectors as input, but according to the special requirements, the function should return an instance with any required initialization. 
# Alternatively, perhaps the vectors are generated inside the my_model_function. Let me see. The original code does:
# vectors = torch.randn(1000, 128)
# So to make the code self-contained, the my_model_function can generate vectors. But according to the structure, the functions must not have test code. The my_model_function should return an instance of MyModel, which requires vectors. So maybe the function can generate vectors on the fly. 
# Wait the function my_model_function must return an instance of MyModel. So perhaps:
# def my_model_function():
#     vectors = torch.randn(1000, 128)
#     return MyModel(vectors)
# But in the original code, vectors are passed to the model's __init__. That's acceptable. 
# The GetInput function must return a random tensor of the correct shape. The original input is torch.randint(1, 1000, (100,30)), which is of shape (100,30), dtype long. So:
# def GetInput():
#     return torch.randint(1, 1000, (100, 30), dtype=torch.long)
# Wait the input shape comment at the top should be:
# # torch.rand(B, L, dtype=torch.long) ← since the input is integer indices.
# So the input is of shape (B, L), where B is batch size and L is sequence length. 
# Now, the special requirements mention that if the issue describes multiple models being compared, they should be fused into a single MyModel with submodules and comparison logic. 
# Wait the issue's problem is comparing different loading methods (m1, m2, m3) which are instances of the same Model class. So perhaps the MyModel should encapsulate both models (like m1 and m2) as submodules, and in its forward, compute both outputs and compare them?
# Alternatively, since the problem arises from the model's structure (two embeddings sharing vectors), the MyModel is the original model. The comparison is done externally, but according to the requirements, if the issue discusses multiple models (like different loading paths), they should be fused. 
# Hmm, the user's code has three different loading methods (m1, m2, m3) which are different instances of the same Model class. The problem is that their outputs differ. The MyModel should encapsulate the scenario where the loading order affects the output. But how to represent that in a single model?
# Alternatively, the MyModel should include both loading methods as submodules. For example, have two submodules, one loaded in each way, and the forward returns their outputs for comparison. But that might be overcomplicating.
# Alternatively, the MyModel is the original model, and the code's structure is as per the original code. The user's issue is about the model's behavior when loaded in different ways. Since the problem is about the model's structure (two embeddings), the MyModel is just the original model. The GetInput function provides the input, and the model's forward is as before. 
# The special requirement says that if the issue describes multiple models (e.g., ModelA, ModelB) compared, they must be fused. But in this case, the models are instances of the same class, just loaded differently. So perhaps the fusion isn't needed here. The problem is within the same model's structure, so the MyModel is the original model. 
# Therefore, the code structure would be as follows:
# The input shape comment is torch.rand(B, L, dtype=torch.long).
# The MyModel class has the two embeddings and linear layer as in the original code.
# The my_model_function generates the vectors and returns the model.
# The GetInput returns the random indices tensor.
# Now, considering the comment's explanation, to fix the problem, the second embedding should use a deep copy of vectors. But the issue's problem is the bug scenario, so the generated code should reproduce the bug. Therefore, the code should have the two embeddings initialized without the deep copy, as in the original code. 
# So the code should be:
# Wait, but in the original code, the vectors are passed to the model's __init__. The my_model_function creates the vectors and passes them. That's correct. 
# Now, the problem's special requirement says that if there are multiple models compared (like the different loading methods), they should be fused. However, in this case, the different loading methods are instances of the same model class, so perhaps the fusion isn't required here. The MyModel is just the original model. 
# The user's issue's problem arises from the two embeddings sharing the same vectors (without deep copy), leading to the described behavior. The code above replicates that scenario. 
# Therefore, this should be the correct code.
# </think>
# ```python
# # torch.rand(B, L, dtype=torch.long)
# import torch
# import torch.nn as nn
# class MyModel(nn.Module):
#     def __init__(self, vectors):
#         super(MyModel, self).__init__()
#         self.emb = nn.Embedding.from_pretrained(vectors, freeze=False, padding_idx=0)
#         self.emb2 = nn.Embedding.from_pretrained(vectors, freeze=False, padding_idx=0)  # Shares same vectors as emb
#         self.linear = nn.Linear(128, 2)
#         
#     def forward(self, s):
#         # Only uses first embedding in forward pass
#         s = self.emb(s)
#         s = self.linear(s)
#         s = s.sum(dim=1)
#         return s
# def my_model_function():
#     vectors = torch.randn(1000, 128)  # Random initial vectors for reproducibility
#     return MyModel(vectors)
# def GetInput():
#     # Generates random indices within the embedding's vocabulary range
#     return torch.randint(1, 1000, (100, 30), dtype=torch.long)
# ```