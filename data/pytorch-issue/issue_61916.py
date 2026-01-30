layer_to_sqnr_dict = get_layer_sqnr_dict(float_model, equalized_model, input)
eq_qconfig_dict = get_equalization_qconfig_dict(layer_to_sqnr_dict, equalized_model, num_layers_to_equalize)