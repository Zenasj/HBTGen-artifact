import torch

add_one = torch.jit.load('/tmp/callgrind__1730858981__c5f7acde-f870-42ed-bbe2-8e7dd1ba51d7/data/add_one.pt')
with open('/tmp/callgrind__1730858981__c5f7acde-f870-42ed-bbe2-8e7dd1ba51d7/data/k.pkl', 'rb') as f:
    k = pickle.load(f)
import sys
sys.path.append('/data/users/huydo/github/pytorch/test/benchmark_utils')
from test_benchmark_utils import MyModule

with torch.serialization.safe_globals([MyModule]):
    model = torch.load('/tmp/callgrind__1730858981__c5f7acde-f870-42ed-bbe2-8e7dd1ba51d7/data/model.pt')