# Some basic setup:
# Setup detectron2 logger
import detectron2
from detectron2.utils.logger import setup_logger
setup_logger()

# import some common libraries
import numpy as np
import cv2
import os
import random
from matplotlib import pyplot as plt

# import some common detectron2 utilities
from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog
from detectron2.structures import BoxMode
from detectron2.data.datasets import register_coco_instances
from detectron2.data.catalog import DatasetCatalog
from detectron2.engine import HookBase

register_coco_instances("boat_train", {}, "/home/Documents/train/instances.json", "/home/Documents/train")
register_coco_instances("boat_val", {}, "/home/Documents/val/instances.json", "/home/Documents/val")

from detectron2.engine import DefaultTrainer
from detectron2.engine import TrainerBase

#Specify Model yaml & weights to grab
cfg = get_cfg()
cfg.merge_from_file(model_zoo.get_config_file("COCO-Detection/faster_rcnn_R_50_C4_1x.yaml"))
cfg.MODEL.WEIGHTS = "/home/svidelock/source/Detectron/model_final_721ade.pkl" # Let training initialize from model zoo 

#Spcify DIR for output, if not specified will create "output" DIR
# cfg.OUTPUT_DIR = '/home/svidelock/source/Detectron/HyperParamDetectron/output2/'

#Specify Datasets
cfg.DATASETS.TRAIN = ("boat_train",) #list of the pre-computed proposal files for trianing
cfg.DATASETS.TEST = ("boat_val",) #validation set

#Hyperparams
cfg.SOLVER.IMS_PER_BATCH = 2 #means that in 1 iteration the model sees 2 images 
cfg.SOLVER.BASE_LR = 0.02 #learning rate

#Some other configurable items
cfg.DATALOADER.NUM_WORKERS = 2 # depends on harware config ... 
# cfg.SOLVER.WARMUP_ITERS = 1000 #constant learning rate
# cfg.SOLVER.STEPS = (1000, 1500) #Decaying learning rate
# cfg.SOLVER.GAMMA = 0.001 # The iteration number to decrease learning rate by GAMMA
cfg.SOLVER.MAX_ITER = 500 # Model will stop after this many iterations
cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 128 #look into
cfg.MODEL.ROI_HEADS.NUM_CLASSES = 1  # only has one class (boat)

#specify if CPU Training
cfg.MODEL.DEVICE='cpu'#cpu training

#Checkpoint/ValidationSet Params
cfg.TEST.EVAL_PERIOD = 20 # Tests validation set every 20 itterations
cfg.SOLVER.CHECKPOINT_PERIOD = cfg.TEST.EVAL_PERIOD #saves a checkpoint model each time we validate 

os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)
trainer = DefaultTrainer(cfg)
trainer.resume_or_load(resume=False)
trainer.train()