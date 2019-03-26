import argparse
import torchfile
from Criterion import Criterion
import torch

parser = argparse.ArgumentParser()

parser.add_argument("-i", "--path_input")
parser.add_argument("-t", "--path_target")
parser.add_argument("-ig", "--grad_input")

args = parser.parse_args()
path_input = args.path_input
path_target = args.path_target
grdi_file = args.grad_input

input_bin = torchfile.load(path_input)
target_bin = torchfile.load(path_target)-1

# print(input_bin.shape)
# print(target_bin.shape) 
# print(input_bin)
try:
	input_bin = torch.from_numpy(input_bin)
except:
	pass
try:
	target_bin = torch.from_numpy(target_bin).long()
	target_bin=target_bin.reshape(target_bin.shape[0])
except:
	pass

avg_loss = Criterion.forward(input_bin,target_bin)
print(avg_loss)
grad_i = Criterion.backward(input_bin,target_bin)/input_bin.shape[0]
torch.save(grad_i,grdi_file)