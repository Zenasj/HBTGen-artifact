import torch
import torchvision
import torch.multiprocessing as mp


def test(m):
    # m.share_memory()

    print('3')
    t = torch.ones((1, 3, 224, 224), dtype=torch.float32)
    print('4')
    with torch.no_grad():
        o = m(t)
    print('fun for all')


def run_forward_local():
    print('Start forward_local')
    torch.set_num_threads(1)
    m = torchvision.models.resnet18()
    t = torch.ones((1, 3, 224, 224), dtype=torch.float32)
    print('Start FW')
    with torch.no_grad():
        o = m(t)
        print('End FW')

    print('End running')


def run_forward_in_child_process_with_model():
    print('Start')
    torch.set_num_threads(1)
    m = torchvision.models.resnet18()
    p = mp.Process(target=run_forward_local, args=(m,))
    p.start()
    p.join()


def run_forward_in_child_process():
    print('Start')
    p = mp.Process(target=run_forward_local)
    p.start()
    p.join()


if __name__ == "__main__":
    # run forward local and then fork and run again
    run_forward_local()
    run_forward_in_child_process()
    print("END ALL")