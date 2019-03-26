config = 'Files/modelConfig_1.txt'
i  = 'Files/input_sample_1.bin'
og = 'gradOutput_sample_1.bin'
o  = 'Files/output_sample_1.bin'
ow = 'Files/gradW_sample_1.bin'
ob = 'Files/gradB_sample_1.bin'
ig = 'Files/gradCriterionInput_sample_1.bin'
import torchfile
import torch
import Model
from Linear import LinearLayer
from ReLu import ReLu
from Criterion import Criterion
from Convolution import ConvolutionLayer, FlattenLayer
import numpy as np

import argparse
parser = argparse.ArgumentParser()
parser.add_argument('-config')
parser.add_argument('-i' )
parser.add_argument('-og')
parser.add_argument('-o' )
parser.add_argument('-ob')
parser.add_argument('-ow')
parser.add_argument('-ig')
args = parser.parse_args()
# print(args.og)
with open(args.config) as f:
    content = f.read().splitlines()
print(content)

No_of_layers=int(content[0])
print(No_of_layers)

model_one = Model.Model()
for i in range(No_of_layers):
    print(content[i+1])
    if content[i+1][0:4]=="relu" :
        print("RElu LAyer Found")
        model_one.addLayer(Relu())
    elif content[i+1][0:6]=="linear" :
        print("Linear LAyer Found")
        x=content[i+1][6:]
        x = [int(i) for i in x.split()]
        model_one.addLayer(LinearLayer(x[0],x[1]))

ith_layer=0
yay=0
for i in range(No_of_layers):
    print(content[i+1+No_of_layers])
    if content[i+1][0:4]=="relu" :
        yay=0
    if content[i+1][0:6]=="linear" :
        model_one.layers[i].W = torch.from_numpy(np.array(torchfile.load(content[i+1+No_of_layers])))[ith_layer]
        ith_layer+=1
ith_layer=0
for i in range(No_of_layers):
    print(content[i+1+2*No_of_layers])
    if content[i+1][0:4]=="relu" :
        yay=0
    if content[i+1][0:6]=="linear" :
        model_one.layers[i].B = torch.from_numpy(np.array(torchfile.load(content[i+1+2*No_of_layers]))).reshape(model_one.layers[ith_layer].B.shape[0],1)
        ith_layer+=1

input_bin = args.i
input_bin_data = torchfile.load(input_bin)
print(input_bin_data.shape)
input_x_with_reshape = input_bin_data.reshape(input_bin_data.shape[0],input_bin_data.shape[1]*input_bin_data.shape[2]*input_bin_data.shape[3])
input_x_with_reshape = torch.from_numpy(input_x_with_reshape).double()

final_label_without_prob =  model_one.forward(input_x_with_reshape)

output_file = args.o
torch.save(final_label_without_prob,output_file)

# backward_gradient = Criterion.backward(final_label_without_prob, Train_Label[] )
gradoutput_file=args.og
backward_gradient =torchfile.load(gradoutput_file)
backward_gradient = torch.from_numpy(backward_gradient).double()
gradinput_to_save  =model_one.backward(input_x_with_reshape,backward_gradient,alpha=1e-5)
torch.save(gradinput_to_save,args.ig)

# ith_layer=0
gradW_to_Save=[]
gradB_to_Save=[]
for i in range(No_of_layers):
    if content[i+1][0:4]=="relu" :
        yay=0
    if content[i+1][0:6]=="linear" :
        gradW_to_Save.append(model_one.layers[i].gradW)
        gradB_to_Save.append(model_one.layers[i].gradB.t()[0])
        # ith_layer+=1

gradW_file = args.ow
torch.save(gradW_to_Save,gradW_file)


# gradB_to_Save = torch.from_numpy(np.array(gradB_to_Save)).reshape()

gradB_file = args.ob
torch.save(gradB_to_Save,gradB_file)
print("end")
