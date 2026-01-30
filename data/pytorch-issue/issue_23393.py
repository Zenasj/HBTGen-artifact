with th.no_grad():
    model.weight.set_(((alice_model.weight.data + bob_model.weight.data) / 2).get())