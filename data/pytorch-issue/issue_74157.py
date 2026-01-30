import torch

for param_group in self.param_groups:
            for params in param_group["params"]:
                if params.grad is None:
                    continue
                self.averager.average_parameters(iter(params))

for param_group in param_groups:
                for params in param_group["params"]:
                    if params.grad is None:
                        continue
                    #utils.average_parameters(iter(params), self.process_group)
                    buffer = flatten([params])
                    buffer = utils.average_parameters_v1(buffer, self.process_group)   # one tensor allreduce
                    params.data[:] = buffer.reshape(params.size())

def average_parameters_v1(
    params: torch.Tensor, process_group: dist.ProcessGroup
):
    """
    Averages all the given parameters.
    For allreduce efficiency, all the parameters are flattened into a contiguous buffer.
    Thus, it requires extra memory of the same size as the given parameters.
    """
    group_to_use = process_group if process_group is not None else dist.group.WORLD
    # Do not update any parameter if not in the process group.
    if dist._rank_not_in_group(group_to_use):
        return

    params /= dist.get_world_size(group_to_use)
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    dist.all_reduce(params, group=group_to_use)
    return params

def flatten(tensors, shapes=None, use_cuda=True):
    # init and recover the shapes vec.
    pointers = [0]
    #print("tensors[0]:{}".format(tensors[0]))
    if shapes is not None:
        for shape in shapes:
            pointers.append(pointers[-1] + shape[1])
    else:
        for tensor in tensors:
            pointers.append(pointers[-1] + tensor.nelement())

   # print("pointers[-1]:{}".format(pointers[-1]))
    #print("tensors[0]:{}".format(tensors[0]))
    # flattening.
    vec = torch.empty(
        pointers[-1],
        device=tensors[0].device if tensors[0].is_cuda and use_cuda else "cpu",
    )

    for tensor, start_idx, end_idx in zip(tensors, pointers[:-1], pointers[1:]):
        vec[start_idx:end_idx] = tensor.data.reshape(-1)
    return vec

tensor([0.9738, 1.0157, 1.0600, 0.9754, 0.9919, 1.0374, 0.9209, 0.9164, 1.0044,
        1.0619, 1.0896, 0.9417, 1.0413, 0.9727, 0.9393, 0.9211, 0.8632, 0.9016,
        1.0059, 1.0765, 1.0447, 0.9740, 0.9073, 0.9917, 0.9445, 1.0774, 1.0274,
        0.8604, 0.8904, 1.0549, 0.9674, 1.1516, 0.9195, 0.9927, 1.0441, 0.8129,
        1.0187, 1.0838, 1.0160, 0.9238, 0.9702, 0.9381, 1.0552, 1.0674, 0.9936,
        1.1840, 0.9342, 0.9749, 0.9944, 1.0190, 1.0209, 1.0386, 1.0583, 1.0107,
        0.9963, 0.9700, 0.9643, 0.9277, 1.2861, 1.1200, 0.9999, 1.0114, 1.0278,
        1.0028], device='cuda:0', requires_grad=True)

tensor([0.9860, 1.0382, 1.0458, 0.9118, 0.9486, 1.0590, 0.9629, 0.9933, 0.9850,
        0.9726, 1.1248, 0.9212, 1.0596, 0.9413, 0.9923, 0.8675, 0.8983, 1.0271,
        0.9149, 1.0182, 1.0369, 0.9818, 0.8172, 1.1388, 1.0603, 0.9485, 1.0161,
        0.9089, 0.8104, 1.0318, 1.0468, 1.1091, 0.8808, 1.0963, 1.0705, 0.8089,
        1.0708, 1.0716, 1.0089, 0.9222, 1.0035, 0.8801, 1.0370, 0.9991, 1.0134,
        1.1002, 0.9589, 1.0312, 0.9316, 1.0986, 0.9080, 1.1331, 0.9658, 0.9599,
        0.9377, 0.9335, 0.9586, 0.9666, 1.3671, 1.1227, 0.9948, 1.0301, 0.9848,
        1.1485], device='cuda:0', requires_grad=True)

tensor([0.9799, 1.0269, 1.0529, 0.9436, 0.9703, 1.0482, 0.9419, 0.9548, 0.9947,
        1.0173, 1.1072, 0.9315, 1.0505, 0.9570, 0.9658, 0.8943, 0.8807, 0.9644,
        0.9604, 1.0473, 1.0408, 0.9779, 0.8623, 1.0652, 1.0024, 1.0129, 1.0218,
        0.8847, 0.8504, 1.0433, 1.0071, 1.1304, 0.9002, 1.0445, 1.0573, 0.8109,
        1.0447, 1.0777, 1.0125, 0.9230, 0.9869, 0.9091, 1.0461, 1.0332, 1.0035,
        1.1421, 0.9466, 1.0030, 0.9630, 1.0588, 0.9644, 1.0859, 1.0121, 0.9853,
        0.9670, 0.9518, 0.9615, 0.9471, 1.3266, 1.1213, 0.9974, 1.0207, 1.0063,
        1.0756], device='cuda:0', requires_grad=True)