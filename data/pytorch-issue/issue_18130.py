import torch
import numpy as np
import random

def get_sampled_loader(cf, test_set):
        no_of_samples  = len(test_set)
        sample_idx = np.random.permutation(np.arange(0, no_of_samples))[:cf.no_of_sampled_data]                     
        if len(sample_idx) ==0:  
            exit('exiting function get_the_sampler(), sample_idx size is 0')    
        my_sampler = torch.utils.data.sampler.SubsetRandomSampler(sample_idx)  
        test_loader = torch.utils.data.DataLoader(test_set, batch_size=cf.batch_size_test,
                          shuffle= False, num_workers=cf.num_workers, sampler=my_sampler)
        return test_loader