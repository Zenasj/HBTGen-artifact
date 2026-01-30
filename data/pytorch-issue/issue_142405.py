[{elem1, elem2, subclass_2}, {elem1, elem2, subclass_1}, {elem1, elem2, nested_subclass_1}]

todo = [tensor1, tensor2, tensor3, tensor4]
d = {a: b for a, b in zip(inner_tensors, todo[-nb_tensor:])}  
todo = todo[nb_tensor:]