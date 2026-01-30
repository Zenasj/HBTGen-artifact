import torch                                                                                                                                 
features = torch.empty(15000000, 45, device="cuda", dtype=torch.long).random_(0, 2**22)                                                                                  
ind=torch.randperm(len(features))                                                                                                            
tmp = torch.ones(15000000,50,device="cuda", dtype=torch.long)
del tmp
features3=features[ind]                                                                                                                      
print( features3[0])
print(features3[14999999] )