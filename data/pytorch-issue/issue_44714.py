import torch                                                                                                                                                                              
                                                                                                                                                
rnd_generator = torch.Generator(device='cuda:0')

print(sorted(torch.utils.data.random_split([1,2,3,4,5,6,7,8,9,0], [8,2], generator=rnd_generator)[0]))