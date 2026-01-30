ab = stable_diffusion_unet_model.forward(...)
a, b = ab.chunk(2)
# a, b have device mps:0
test1 = a - b
# test1 is all 0s
test2 = a - b.clone()
# test2 has what looks like valid output
test3 = a.clone() - b
# test3 is all 0s
test4 = a.cpu() - b.cpu()
# test4 has what looks like valid output, matching test2