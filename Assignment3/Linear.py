import torch
import math
# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
device = torch.device('cpu')

class LinearLayer:
    
    def __init__(self, in_nodes, out_nodes):
        self.output = None
        # self.output = torch.empty(batch_size, out_nodes, dtype=torch.double)
        #batchsize = 1
        # output which is a matrix of size (batch size × number of output neurons)
        sqrt_2_in_nodes = math.sqrt(in_nodes)
        # sqrt_2_in_nodes = math.sqrt(in_nodes/2.0)
        self.W = torch.randn(out_nodes,in_nodes,dtype=torch.double,device=device)/sqrt_2_in_nodes
        # W which is a matrix of size (number of output neurons × number of input neurons)
        self.B = torch.randn(out_nodes,1,dtype = torch.double,device=device)/sqrt_2_in_nodes
        # B which is a matrix of size (number of output neurons × 1)
        self.gradW =torch.empty(out_nodes,in_nodes ,dtype = torch.double,device=device)
        # gradW being the same size is W
        self.gradB = torch.empty(out_nodes,1 ,dtype = torch.double,device=device)
        # gradB also being the same size as B and
        self.gradInput   = None




    def forward(self, input):
        self.output = torch.mm(self.W,input.t()).t() + self.B.t()
        return self.output

    def backward(self,input , gradOutput,alpha):
        # print("in layers abckward\t",gradOutput)
        self.gradInput = torch.mm(gradOutput,self.W)
        self.gradB     =   sum(gradOutput).reshape(self.B.shape)
        self.gradW     =   torch.mm(gradOutput.t(),input)
        self.B -= alpha * self.gradB
        self.W -= alpha * self.gradW

        return self.gradInput 
        
