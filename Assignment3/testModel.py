import argparse
import torch
import torchfile

parser = argparse.ArgumentParser()

parser.add_argument("-modelName", "--model_name")
parser.add_argument("-data", "--data_path")

args = parser.parse_args()
modelName = args.model_name
test_path = args.data_path

model_one=torch.load(modelName)
Test_Data = torchfile.load(test_path)

Test_Data = torch.from_numpy(Test_Data).double()
Test_Data = Test_Data.reshape(Test_Data.shape[0],1,108,108)

_,indices = model_one.forward(Test_Data).max(1) 

torch.save(indices,'testPrediction.bin')
# ind = torch.load('testPrediction.bin')
