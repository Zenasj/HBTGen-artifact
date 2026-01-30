import torch

# instantiate the quantized net (not shown here).

# get one of the conv layers
tmp = model_int8.state_dict()['features.0.weight']
scales = tmp.q_per_channel_scales()
zero_pts = tmp.q_per_channel_zero_points()
axis = tmp.q_per_channel_axis()

# get int repr
tmp_int8 = tmp.int_repr()

# change value (value change is dependent on the int8 value)
tmp_int8[0][0][0][0] = new_value

# how to convert tmp_int8 to torch.qint8 type?
new_tmp = torch._make_per_channel_quantized_tensor(tmp_int8, scales, zero_pts, axis)

# based on the above step:
model_int8.features[0].weight = new_tmp

# shows updated values
print(model_int8.features[0].weight)

# shows old values
print(model_int8.state_dict()['features.0.weight'])

# inference is unchanged compared to original quantized model