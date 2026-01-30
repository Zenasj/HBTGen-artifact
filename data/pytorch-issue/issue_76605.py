with FusionDefinition(fusion) as fd :
    t0 = fd.define_tensor(2, DataType.Half)
    t1 = fd.define_tensor(2, DataType.Double)

    fd.add_input(t0)
    fd.add_input(t1)

    t2 = fd.Ops.add(t0, t1)
    t5 = fd.Ops.relu(t2)

    fd.add_output(t5)