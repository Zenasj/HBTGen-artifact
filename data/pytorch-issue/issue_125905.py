import torch

def test_all(self, epoch):
        self.model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for data, target in self.data_manager.test_loader:
                data, target = data.to(self.device), target.to(self.device)
                output = self.model(data)
                pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
                correct += pred.eq(target.view_as(pred)).sum().item()
                total += target.size(0)

        correct_tensor = torch.tensor(correct).to(self.device)
        total_tensor = torch.tensor(total).to(self.device)

        dist.allreduce(correct_tensor)
        dist.allreduce(total_tensor)

        if self.rank == 0:  
            accuracy = correct_tensor.item() / total_tensor.item()
            self.writer.add_scalar('Accuracy/test', accuracy, epoch)