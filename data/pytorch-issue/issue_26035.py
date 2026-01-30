model = onnx.load("file.onnx")
graph = model.graph
nodes = graph.node

count = 0
for node in nodes:
    if not node.name:
        node.name = "rand_node_name_" + str(count)
        count = count + 1