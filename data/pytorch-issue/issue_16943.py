import torch
import numpy as np

with Pool(self.args.numThreads) as pool:
    for result in list(tqdm(pool.imap(self.executeEpisode, range(self.args.numEps)), total=self.args.numEps, desc='Self-play matches')):
        iterationTrainExamples += result

pi = self.mcts.getActionProb(state, temp=temp)

self.Ps[s], v = self.nnet.predict(state)        # let neural network predict action vector P and state value v

def predict(self, state):
    state = torch.FloatTensor(state.astype(np.float64)).unsqueeze(0).unsqueeze(0)
    state = state.contiguous().cuda()
    self.nnet.eval()
    with torch.no_grad():
        pi, v = self.nnet(state)
    return torch.exp(pi).data.cpu().numpy()[0], v.data.cpu().numpy()[0]

torch.load(filepath, torch.device('cpu'))

import subprocess
import time

for i in range(8):
    subprocess.Popen('python selfplay_client.py')
    time.sleep(1)