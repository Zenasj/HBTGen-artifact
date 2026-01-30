import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class DummyModel(nn.Module):
    def __init__(self, in_dim=10, n_classes=2, model_type='classification'):
        super().__init__()
        self.model_type = model_type
        self.in_dim = in_dim
        self.n_classes = n_classes

        self.classifier = nn.Linear(in_dim, n_classes)

    def forward(self, x):
        return self.classifier(x)

    @torch.no_grad()
    def predict(self, x):
        outputs = self.forward(x)
        if self.model_type == 'regression':
            return outputs
        elif self.model_type == 'classification':
            return self.predict_proba(outputs, compute_outputs=False)
        raise ValueError(f'Param "model_type" ("{self.model_type}") must be one of ["classification", "regression"]')

    @torch.no_grad()
    def predict_proba(self, x, compute_outputs=True):
        '''
        Executed when `model_type="classification"`
        and `self.predict()` is called.
        :param x: Raw inputs or model output logits
        :compute_outputs: If true, computes model output first,
                          otherwise assumes x is already the
                          output of the model
        :return outputs: Raw model output logits
        :return preds: Predicted classes
        :return probs: Predicted probabilities of each class
        '''
        outputs = self.forward(x) if compute_outputs else x # Get model outputs if necessary
        probs = F.softmax(outputs, dim=-1) # Get probabilities of each class
        preds = probs.max(dim=-1)[1] # Get class with max probability
        return outputs, preds, probs

@torch.no_grad()
def get_all_predictions(model, dataloader, device):
    '''
    Make predictions on entire dataset using model's own
    `predict()` method and return raw outputs and additionally
    class predictions and probabilities if it's a classification task
    '''
    num_batches, num_examples = len(dataloader), len(dataloader.dataset)
    batch_size = np.ceil(num_examples / num_batches).astype(int)
    model.eval()

    outputs_hist, preds_hist, probs_hist = [], [], []
    for batch_idx, batch in enumerate(dataloader):
        x = batch[0].to(device) # Only need inputs for making predictions
        outputs, preds, probs = model.predict(x)
#         outputs, preds, probs = model(x), [], [] # <------- Replacing above line with this works.
        outputs_hist.extend(outputs)
        if model.model_type == 'classification': # Get class predictions and probabilities
            preds_hist.extend(preds)
            probs_hist.extend(probs)

        # Print 50 times in a batch
        if (batch_idx+1) // num_batches % 2 == 0:
            print('{}/{} ({:.0f}%) complete.'.format(
                  (batch_idx+1) * batch_size, num_examples,
                  100. * (batch_idx+1) / num_batches))

    return outputs_hist, preds_hist, probs_hist

model = DataParallel(DummyModel(DIM, N_CLASSES).to(DEVICE), device_ids=DEVICE_IDS)
outputs, preds, probs = get_all_predictions(model, dataloader, DEVICE)