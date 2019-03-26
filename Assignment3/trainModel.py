import argparse
import os
from shutil import copy2
import torchfile
import Model
from Linear import LinearLayer 
from ReLu import ReLu
from Convolution import ConvolutionLayer,FlattenLayer
from Criterion import Criterion
import torch
import final

# suffix='.py'

parser = argparse.ArgumentParser()

parser.add_argument("-modelName", "--model_name")
parser.add_argument("-data", "--data_path")
parser.add_argument("-target", "--target_path")

args = parser.parse_args()
model_name = args.model_name
data_path = args.data_path
labels_path = args.target_path

# dir_name = modelName.rsplit(suffix,1)[0]

try:
	os.mkdir(model_name)
	print("Directory created")
except:
	print("Directory already exists")

model_one = torch.load(model_name)
Train_Data = torchfile.load(data_path)
Train_Label = torchfile.load(labels_path)

# model_one=Model.Model()
# model_one.addLayer(ConvolutionLayer( (1,108,108) , 12 , 15, 6))
# model_one.addLayer(ReLu())
# model_one.addLayer(ConvolutionLayer( (15,17,17) , 5 , 9, 3)) #9,5,5
# model_one.addLayer(FlattenLayer())
# model_one.addLayer(ReLu())
# model_one.addLayer(LinearLayer(225,90))
# model_one.addLayer(ReLu())
# model_one.addLayer(LinearLayer(90,18))
# model_one.addLayer(ReLu())
# model_one.addLayer(LinearLayer(18,6))


model_one=final.train(model_one,Train_Data,Train_Label)

torch.save(model_one,'./'+model_name+'/'+model_name)