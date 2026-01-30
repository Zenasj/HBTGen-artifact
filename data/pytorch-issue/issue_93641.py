from torchbenchmark.models.resnet50 import Model as R50Model
import torchdynamo
class MyR50(R50Model):
    
    @torchdynamo.optimize("aot_nvfuser")
    def train(self, niter=1):
        for _ in range(niter):
            self.optimizer.zero_grad()
            for data, target in zip(self.real_input, self.real_output):
                pred = self.model(data)
                self.loss_fn(pred, target).backward()
                self.optimizer.step()

r50 = MyR50('train', 'cuda')
r50.train()

from torchbenchmark.models.resnet50 import Model as R50Model
import torchdynamo
class MyR50(R50Model):
    
    @torchdynamo.optimize("aot_nvfuser")
    def train(self, niter=1):
        for _ in range(niter):
            #self.optimizer.zero_grad()
            for data, target in zip(self.real_input, self.real_output):
                pred = self.model(data)
                self.loss_fn(pred, target).backward()
                self.optimizer.step()

r50 = MyR50('train', 'cuda')
r50.train()