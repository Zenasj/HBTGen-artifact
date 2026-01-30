import json
import matplotlib.pyplot as plt
import numpy as np
import os
import re
import PIL.Image
from sklearn.metrics import accuracy_score
import torch
import torchvision.models as models
from torchvision.models import resnet50, ResNet50_Weights
import torchvision.transforms as transforms
from tqdm import tqdm
import pandas as pd
from torch.utils.data import RandomSampler
    
from shared_interest.datasets.imagenet import ImageNet
from shared_interest.shared_interest import shared_interest
from shared_interest.util import flatten, binarize_std
from interpretability_methods.vanilla_gradients import VanillaGradients

def load_model_from_pytorch(architecture, pretrained):
    model = models.__dict__[architecture](weights=ResNet50_Weights.IMAGENET1K_V1)
    #model = models.__dict__[architecture](pretrained=pretrained)
    return model

model = load_model_from_pytorch('resnet50', pretrained=True)
#device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
device = 'cpu'
model = model.eval().to(device)

imagenet_dir = '/Users/davidlaxer/ImageNet/ILSVRC'
image_dir = os.path.join(imagenet_dir, 'Data/CLS-LOC/train')
annotation_dir = os.path.join(imagenet_dir, 'Annotations/CLS-LOC/train')

df = pd.read_fwf('/Users/davidlaxer/ImageNet/LOC_synset_mapping.txt', sep=' ', index_col=False, header=None, names=['WordNetId', 'Class'], set_index='WordNetId')
df['row_num'] = np.arange(len(df))

# ImageNet transforms.
image_transform = transforms.Compose([transforms.Resize(256),
                                      transforms.CenterCrop(224),
                                      transforms.ToTensor(),
                                      transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                                                           std=[0.229, 0.224, 0.225]),
                                     ])

ground_truth_transform = transforms.Compose([transforms.ToPILImage(),
                                             transforms.Resize(256, PIL.Image.NEAREST),
                                             transforms.CenterCrop(224),
                                             transforms.ToTensor()])

reverse_image_transform = transforms.Compose([transforms.Normalize(mean=[0, 0, 0], 
                                                                   std=[4.3668, 4.4643, 4.4444]),
                                              transforms.Normalize(mean=[-0.485, -0.456, -0.406], 
                                                                   std=[1, 1, 1]),
                                              transforms.ToPILImage(),])

dataset = ImageNet(image_dir, annotation_dir, image_transform, ground_truth_transform)
dataloader = torch.utils.data.DataLoader(dataset, batch_size=2, shuffle=True, num_workers=0, pin_memory=True)

for image, ground_truth, label in dataset:
    if ground_truth == None:
        continue
    fig, ax = plt.subplots(nrows=1, ncols=2)
    ax[0].imshow(reverse_image_transform(image))
    ax[1].imshow(ground_truth)
    type(image)
    break

saliency_method = VanillaGradients(model)

def run(model, dataloader, saliency_method, stop_after=None):
    """
    Runs the model through the data in the dataloader and computes the 
    predictions, saliency, and Shared Interest scores.
    
    Args:
    model: pytorch model to evaluate.
    dataloader: dataloader to evaluate the model on. Should output an image,
        ground truth, and label for each index.
    saliency_method: the InterpretabilityMethod to use to compute Shared
        Interest.
    stop_after: optional integer that determins when to stop the evaluation.
        Used here for efficiency in the notebook. If None, entire process will
        run.
    """
    accuracy = 0
    total_shared_interest_scores = {'iou_coverage': np.array([]),
                                    'ground_truth_coverage': np.array([]),
                                    'saliency_coverage': np.array([]),}
    total_predictions = np.array([])
    total_saliency_masks = np.array([[[]]])
    for i, (images, ground_truth, labels) in enumerate(tqdm(dataloader)):
        with torch.no_grad():
            images = images.to(device, torch.float32)
            ground_truth = ground_truth.numpy()
            print(labels)
            df1 = pd.DataFrame(labels, columns=['WordNetId'])
            df2 = df.merge(df1)['row_num']
            labels = df2.to_numpy()
        
            # Compute model predictions
            output = model(images)
            predictions = torch.argmax(output, dim=1).detach().cpu().numpy()
            total_predictions = np.concatenate((total_predictions, predictions))
            
            # Update metrics
            print(accuracy_score(labels, predictions))
            print("Labels:\t\t", labels)
            #print("Output:\t\t", output)
            print("Predictions:\t", predictions, "\n")
            accuracy += accuracy_score(labels, predictions)
        
        # Compute saliency
        saliency = flatten(saliency_method.get_saliency(images))
        saliency_masks = binarize_std(saliency)
        if i == 0: 
            total_saliency_masks = saliency_masks
        else:
            total_saliency_masks = np.concatenate((total_saliency_masks, saliency_masks))
        
        # Compute Shared Interest scores
        for score in total_shared_interest_scores:
            shared_interest_scores = shared_interest(ground_truth, saliency_masks, score=score)
            total_shared_interest_scores[score] = np.concatenate((total_shared_interest_scores[score], shared_interest_scores))

        # Stop early for this example notebook
        if stop_after and stop_after == i:
            break
    accuracy /= i + 1
    print('Accuracy: %.2f' %accuracy)
    return total_saliency_masks, total_shared_interest_scores, total_predictions

saliency_masks, shared_interest_scores, predictions = run(model, dataloader, saliency_method, stop_after=10)

py
df2 = df.merge(df1)['row_num']
labels = df2.to_numpy()