import torch.utils.model_zoo as model_zoo
import torch.multiprocessing as mp

if __name__ == '__main__':
	mp.set_start_method('spawn')