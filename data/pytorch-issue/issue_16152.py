import torch.utils.model_zoo as model_zoo
RESTORE_FROM = "http://vllab.ucmerced.edu/ytsai/"\
               "CVPR18/DeepLab_resnet_pretrained_init-f81d91e8.pth"
model_zoo.load_url(RESTORE_FROM)