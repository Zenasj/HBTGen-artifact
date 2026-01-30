import torch
import argparse
import torchvision.models as models
import torch.backends.cudnn as cudnn
import timeit
import numpy as np
import torch.nn as nn

def printStats(graphName, timings, batch_size):
    times = np.array(timings)
    steps = len(times)
    speeds = batch_size / times
    time_mean = np.mean(times)
    time_med = np.median(times)
    time_99th = np.percentile(times, 99)
    time_std = np.std(times, ddof=0)
    speed_mean = np.mean(speeds)
    speed_med = np.median(speeds)

    msg = ("\n%s =================================\n"
            "batch size=%d, num iterations=%d\n"
            "  Median FPS: %.1f, mean: %.1f\n"
            "  Median latency: %.6f, mean: %.6f, 99th_p: %.6f, std_dev: %.6f\n"
            ) % (graphName,
                batch_size, steps,
                speed_med, speed_mean,
                time_med, time_mean, time_99th, time_std)
    print(msg)


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description="Run inference on a model with random input values")
    parser.add_argument('--model', default='resnet50', type=str, help='model architecture: resnet18, resnet34, resnet50, resnet101, mobilenet, mobilnetv2')
    parser.add_argument('--jit', action='store_true', help='run jit mode')
    parser.add_argument('--gpu', default=0, type=int, help='GPU id to use.')
    parser.add_argument('--fp16', action='store_true', help='Run inference with half precision float')
    parser.add_argument("--batch_size", type=int, default=1, help="Batch size (default=1)")
    parser.add_argument("--iter", default=100, type=int, help="Number of iteration loops")
    args = parser.parse_args()

    if args.model:
        model_switch = {
            "resnet18" : models.resnet18(),
            "resnet34" : models.resnet34(),
            "resnet50" : models.resnet50(),
            "resnet101" : models.resnet101(),
            "resnet152" : models.resnet152(),
            "densenet" : models.DenseNet(),
        }
        # Creating model with random weights
        print("Creating model '{}'".format(args.model))

        model = model_switch.get(args.model, "Invalid")
        if model == "Invalid":
            raise Exception("Invalid model")
    
    modules=list(model.children())[:-1]
    model=nn.Sequential(*modules)
    model = model.eval()

    if args.jit:
        print("Generate traced model...")
        example_input = torch.rand(args.batch_size, 3, 224, 224, dtype=torch.float)
        model = torch.jit.trace(model, example_input,
            check_trace=True,
            check_tolerance=1e-05,
            optimize=True,
            )

    # Create graph on GPU if CUDA is available
    if torch.cuda.is_available():
        # Select GPU to use
        device = torch.device('cuda', args.gpu)
        # Print out some status
        print("Cuda DNN version=", torch.backends.cudnn.version())
        print("Current GPU id, GPU count=", torch.cuda.current_device(), torch.cuda.device_count())
        print("Cuda compute capability=", torch.cuda.get_device_capability(device))
        print("Cuda device name=", torch.cuda.get_device_name(device))
        # Enable CuDNN autotune for better performance (with fixed inputs)
        cudnn.benchmark = True
    else:
        # raise Exception("No cuda available.")
        device = torch.device('cpu')

    # Set input shape (NCWH)
    input_shape = (args.batch_size, 3, 224, 224)
    # Create random input tensor of certain size
    torch.manual_seed(12345)
    input_t = torch.rand(input_shape, dtype=torch.float)

    if args.fp16:
        # Cast model and data to half precision
        model = model.half()
        input_t = input_t.half()

    # Copy model and input to device
    model = model.to(device)
    input_t = input_t.to(device)
    timings=[]
    with torch.no_grad():
        for i in range(args.iter):
            start_time = timeit.default_timer()
            features = model(input_t)
            torch.cuda.synchronize()
            end_time = timeit.default_timer()
            if i ==0:
                continue
            else:
                timings.append(end_time - start_time)
            #print("Iteration {}: {:.6f} s".format(i, end_time - start_time))
    
    print("Input shape:", input_t.size())
    print("Output features size:", features.size())

    printStats(args.model, timings, args.batch_size)