from torch.autograd import Variable
import torch.onnx
import torchvision
from Networks.LSQ_layer import Net
from Networks.utils import define_args, save_weightmap, first_run, \
                           mkdir_if_missing, Logger, define_init_weights,\
                           define_scheduler, define_optim, AverageMeter

dummy_input = Variable(torch.randn(1,3,256,512)).cuda()
parser = define_args()
args = parser.parse_known_args()[0]  

model = Net(args)
define_init_weights(model, args.weight_init)
checkpoint = torch.load("model_best_epoch_204.pth.tar")
model.load_state_dict(checkpoint['state_dict'])
model = model.cuda()

torch.onnx.export(model, dummy_input, "LaneDetection.onnx", verbose=True)

def _iter_filter(condition, allow_unknown=False, condition_msg=None,
                 conversion=None):#lulu change allow_unknown=False to True
    def _iter(obj):
        if conversion is not None:
            obj = conversion(obj)
        if condition(obj):
            yield obj
        elif obj is None:
            return
        elif isinstance(obj, (list, tuple)):
            for o in obj:
                for var in _iter(o):
                    yield var
        elif allow_unknown:
            yield obj
        else:
            raise ValueError("Auto nesting doesn't know how to process "
                             "an input object of type " + torch.typename(obj) +
                             (". Accepted types: " + condition_msg +
                              ", or lists/tuples of them"
                              if condition_msg else ""))

    return _iter