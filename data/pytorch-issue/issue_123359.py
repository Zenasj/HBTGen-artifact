def forward(inputs):
    __compiled_fn_0 = ...  # The actual function needs to be provided
    graph_out_0 = __compiled_fn_0(inputs)  # clears inputs
    temp_list = []
    temp_list.append(graph_out_0[0])
    inputs[4].grad = graph_out_0[1]  # inputs is empty, index error
    inputs[7].grad = graph_out_0[2]
    inputs[8].grad = graph_out_0[3]
    inputs[9].grad = graph_out_0[3]
    del graph_out_0
    return temp_list

def forward(inputs):
    __compiled_fn_0 = ...  # The actual function needs to be provided
    inputs_ref_1 = inputs[9]
    inputs_ref_2 = inputs[4]
    inputs_ref_3 = inputs[8]
    inputs_ref_4 = inputs[7]
    graph_out_0 = __compiled_fn_0(inputs)
    temp_list = []
    temp_list.append(graph_out_0[0])
    inputs_ref_2.grad = graph_out_0[1]
    inputs_ref_4.grad = graph_out_0[2]
    inputs_ref_3.grad = graph_out_0[3]
    inputs_ref_1.grad = graph_out_0[3]
    del graph_out_0
    return temp_list