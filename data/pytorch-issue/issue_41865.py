from concurrent.futures import ThreadPoolExecutor

def run_model():
  ...

executor = ThreadPoolExecutor(max_workers=10)
futures = []
for i in range(1000):
    future = executor.submit(run_model)
    futures.append(future)

for future in futures:
    future.result()

import torch
import torch.nn as nn
from concurrent.futures import ThreadPoolExecutor
import time
import argparse
from tabulate import tabulate
from transformers import DistilBertModel, DistilBertTokenizer


def run_model(tokenizer, model, txt):
    input_ids = torch.tensor(tokenizer.encode(txt, add_special_tokens=True)).unsqueeze(0)
    outputs = model(input_ids)
    return outputs

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--quantize', action='store_true')
    parser.add_argument('--threads', nargs='+', type=int, default=[1, 2, 4, 8, 16])
    args = parser.parse_args()

    print ("Instantiating tokenizer")
    tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
    print ("Instantiating model")
    model = DistilBertModel.from_pretrained('distilbert-base-uncased')

    if args.quantize:
        model = torch.quantization.quantize_dynamic(model, {torch.nn.Linear}, dtype=torch.qint8)

    times = []
    for threads in args.threads:
        print ("Testing with %d threads" % threads)
        if threads == 0:
            t0 = time.time()
            run_model(quantize=args.quantize)
            t1 = time.time()
        else:
            executor = ThreadPoolExecutor(max_workers=threads)
            futures = []
            t0 = time.time()
            for i in range(1000):
                txt = 'this is a test %i' % i
                future = executor.submit(run_model, tokenizer, model, txt)
                futures.append(future)

            for future in futures:
                future.result()
            t1 = time.time()
            executor.shutdown()
        times.append(t1 - t0)
    header = ["Threads"] + args.threads
    table = [["Time"] + times, ["Rel"] + [float(t)/times[0] for t in times]]
    print("Times for quant={}".format(args.quantize))
    print(tabulate(table, header))

import torch
import torch.quantization
import torch.nn as nn
import copy
import os
import time
import timeit
from concurrent.futures import ThreadPoolExecutor


class linear_for_demonstration(nn.Module):
  def __init__(self, IN, OUT):
     super(linear_for_demonstration,self).__init__()
     self.linear = nn.Linear(IN, OUT)

  def forward(self, inputs):
     return self.linear(inputs)


torch.manual_seed(29592)  # set the seed for reproducibility

#shape parameters
IN=1024
N=20
OUT=1024
WORKERS=4

# random data for input
inputs = torch.randn(N, IN)


 # here is our floating point instance
float_linear = linear_for_demonstration(IN, OUT)

# this is the call that does the work
quantized_linear = torch.quantization.quantize_dynamic(
    float_linear, {nn.Linear}, dtype=torch.qint8
)
def run_float_model():
    float_linear.forward(inputs)
    #pass
def run_quant_model():
    quantized_linear.forward(inputs)
    #pass


executor = ThreadPoolExecutor(max_workers=WORKERS)

start = time.time()
futures = []
for i in range(10000):
    future = executor.submit(run_float_model)
    futures.append(future)

for future in futures:
    future.result()

end = time.time()
print("float time with {} workers: ".format(WORKERS))
print(end - start)

start = time.time()
futures = []
for i in range(10000):
    future = executor.submit(run_quant_model)
    futures.append(future)

for future in futures:
    future.result()

end = time.time()
print("quant time with {} workers: ".format(WORKERS))
print(end - start)

start = time.time()
for _ in range(100):
    float_linear.forward(inputs)
end = time.time()
print("float time: ")
print(end - start)

start = time.time()
for _ in range(100):
    quantized_linear.forward(inputs)
end = time.time()
print("quantize_time time: ")
print(end - start)

# show the changes that were made

print('Here is the f oating point version of this module:')
print(float_linear)
print('')
print('and now the quantized version:')
print(quantized_linear)

def print_size_of_model(model, label=""):
    torch.save(model.state_dict(), "temp.p")
    size=os.path.getsize("temp.p")
    print("model: ",label,' \t','Size (KB):', size/1e3)
    os.remove('temp.p')
    return size

# compare the sizes
f=print_size_of_model(float_linear,"fp32")
q=print_size_of_model(quantized_linear,"int8")
print("{0:.2f} times smaller".format(f/q))