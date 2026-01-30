import torch
import torchvision
import numpy as np

def train(self, num_epoch = 10, gpu = True):
        
        if gpu : 
            CUDA_LAUNCH_BLOCKING="1"

            #torch.set_default_tensor_type(torch.FloatTensor) 
            model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True)
            use_cuda = torch.cuda.is_available()
            device = torch.device("cuda:0" if use_cuda else "cpu")
            model.to(device)
            if self.multi_object_detection == False : 
                num_classes = 2 # ['Tool', 'background']
            else : 
                print("need to set a multi object detection code")

            in_features = torch.tensor(model.roi_heads.box_predictor.cls_score.in_features, dtype = torch.int64).to(device)
            print("in_features = {}".format(in_features))
            model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
            print( "model.roi_heads.box_predictor {}".format( model.roi_heads.box_predictor))
            
            model_parameters = filter(lambda p: p.requires_grad, model.parameters())
            #params = sum([np.prod(p.size()) for p in model_parameters])
            params = [p for p in model.parameters() if p.requires_grad]

            
            optimizer = torch.optim.SGD(params, lr=0.001, momentum=0.9, weight_decay=0.0005)
            lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)
            gc.collect()
            num_epochs = 5
            FILE_model_dict_gpu = "model_state_dict__gpu_lab2_and_lab7_5epoch.pth"
            list_of_list_losses = []
            print("device = ", device)
            
            if (self.data_loader.dataset) == None :
                self.build_dataloader(device)
            
            for epoch in tqdm(range(num_epochs)):

                # Train for one epoch, printing every 10 iterations
                train_his_, list_losses, list_losses_dict = train_one_epoch(model, optimizer, self.data_loader, device, epoch, print_freq=10)
                lr_scheduler.step()

def train_one_epoch(model, optimizer, data_loader, device, epoch, print_freq):

    model.train()
    metric_logger = utilss.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', utilss.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    header = 'Epoch: [{}]'.format(epoch)
    list_losses = []
    list_losses_dict = []
    for i, values in tqdm(enumerate(metric_logger.log_every(data_loader, print_freq, header))):
        images, targets = values
        for image in images : 
            print("before the to(device) operation, image.is_cuda = {}".format(image.is_cuda))
        images = list(image.to(device, dtype=torch.float) for image in images)
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
        #images = [image.cuda() for image in images]
        for image in images : 
            print(image)
            print("after the to(device) operation, image.is_cuda = {}".format(image.is_cuda))
        for target in targets :
            for t, dict_value in target.items():
                print("after the to(device) operation, dict_value.is_cuda = {}".format(dict_value.is_cuda))

        print("images = {}".format(images))
        print("targets = {}".format(targets))

        # Feed the training samples to the model and compute the losses
        loss_dict = model(images, targets)
        losses = sum(loss for loss in loss_dict.values())

py
model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)