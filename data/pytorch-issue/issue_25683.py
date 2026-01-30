import torch
import numpy as np

def letterbox_image(img, inp_dim):
    '''resize image with unchanged aspect ratio using padding'''
    img_w, img_h = img.shape[1], img.shape[0]
    w, h = inp_dim
    new_w = int(img_w * min(w/img_w, h/img_h))
    new_h = int(img_h * min(w/img_w, h/img_h))
    resized_image = cv2.resize(img, (new_w, new_h),
                               interpolation=cv2.INTER_CUBIC)
    canvas = np.full((inp_dim[1], inp_dim[0], 3), 128)
    canvas[(h-new_h)//2:(h-new_h)//2 + new_h,
           (w-new_w)//2:(w-new_w)//2 + new_w, :] = resized_image
    return canvas


def prep_image(img, inp_dim):
    """
    Prepare image for inputting to the neural network.
    Returns a Variable
    """

    orig_im = img.copy()
    dim = orig_im.shape[1], orig_im.shape[0]
    img = (letterbox_image(orig_im, (inp_dim, inp_dim)))
    img_ = img[:, :, ::-1].transpose((2, 0, 1)).copy()
    img_ = torch.from_numpy(img_).float().div(255.0).unsqueeze(0)
    return img_, orig_im, dim


def Net_MakeDetections(model,frame):
    CUDA = torch.cuda.is_available()
    inp_dim = int(model.net_info["height"])
    assert inp_dim % 32 == 0
    assert inp_dim > 32
    nms_thesh               =       float(0.3)
    classes                 =       open('data/coco.names', "r").read().split("\n")[:-1]
    confidence              =       float(0.3)
    num_classes             =       len(classes)


    img, orig_im, dim = prep_image(frame, inp_dim)
    im_dim = torch.FloatTensor(dim).repeat(1, 2)
    if CUDA:
        im_dim = im_dim.cuda()
        img = img.cuda()
    with torch.no_grad():
        output = model(Variable(img), CUDA)
    output = write_results(output, confidence, num_classes,nms=True, nms_conf=nms_thesh)
    im_dim = im_dim.repeat(output.size(0), 1)
    scaling_factor = torch.min(inp_dim/im_dim, 1)[0].view(-1, 1)
    output[:, [1, 3]] -= (inp_dim -scaling_factor*im_dim[:, 0].view(-1, 1))/2
    output[:, [2, 4]] -= (inp_dim - scaling_factor*im_dim[:, 1].view(-1, 1))/2
    output[:, 1: 5] /= scaling_factor

    for i in range(output.shape[0]):
        output[i, [1, 3]] = torch.clamp(output[i, [1, 3]], 0.0,im_dim[i, 0])
        output[i, [2, 4]] = torch.clamp(output[i, [2, 4]], 0.0,im_dim[i, 1])
    try:
        a = [out.cpu().numpy()[[1, 2, 3, 4, 5, 7]] for out in output if classes[int(out[-1])] in ['person']]
        #if len(a)==0:print(list_number,'---> yolo does not have detections')
    except:
        print(' ----> Exception at yolo_detections()',[int(out[-1]) for out in output])
        a=[]
    return a