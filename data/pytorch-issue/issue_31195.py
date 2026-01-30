import torch.nn as nn

import torch
def main():
	dim = 181;
	input_data = torch.randn(1,1,dim ,dim );
	filter_data = torch.randn(1,1,dim ,dim );	
	torch.nn.functional.conv2d(filter_data, input_data, padding=(int(dim /2 - 1), int(dim /2 - 1)));
if __name__ == "__main__":
	main()