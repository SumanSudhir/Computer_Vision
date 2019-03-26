import torch
import math
import numpy as np
# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
device = torch.device('cpu')

class ConvolutionLayer:
	
	# filter_size = n*n (assumed)
	def __init__(self, in_channels, filter_size, numfilters, stride):   
		# Parameters
		# in_channels - list of 3 elements denoting size of input for convolution layer
		# filter_size - list of 2 elements denoting size of kernel weights for convolution layer
		# numfilters  - number of feature maps (denoting output depth)
		# stride	  - stride to used during convolution forward pass
		self.f_size = filter_size
		self.stride = stride
		self.out_depth = numfilters
		self.out_col = int((in_channels[2] - filter_size)/self.stride + 1)
		self.out_row = int((in_channels[1] - filter_size)/self.stride + 1)
		self.output = None
		for_divide = math.sqrt(in_channels[0]*in_channels[1]*in_channels[2])
		# for_divide = math.sqrt(in_channels[0]*in_channels[1]*in_channels[2]/2.0)
		self.W = torch.randn(numfilters, in_channels[0] , filter_size, filter_size,dtype = torch.double,device=device )/for_divide
		self.B = torch.randn(self.out_depth,dtype = torch.double,device=device)/for_divide
		self.gradW = torch.empty(self.W.shape,dtype = torch.double,device=device)
		self.gradB = torch.empty(self.B.shape,dtype = torch.double,device=device)
		self.gradInput = None
		
		

	def forward(self, X):
		self.output = torch.empty(X.shape[0] ,self.out_depth, self.out_row, self.out_col ,  dtype = torch.double,device=device)
		for i in range(self.out_depth) :
			for j in range(self.out_row):
				for k in range(self.out_col):
					self.output[:,i,j,k]= (self.W[i] * X[:,:,j*self.stride:j*self.stride+self.f_size,k*self.stride:k*self.stride+self.f_size]).sum(1).sum(1).sum(1)
			self.output[:,i,:,:] += self.B[i]
		return self.output

	def backward(self,  input_previous, delta_b_r_c,lrate):
		n = input_previous.shape[0] # batch size


		for i in range(self.W.shape[0]):
			for j in range(self.W.shape[1]):
				for k in range(self.W.shape[2]):
					for l in range(self.W.shape[3]):
						self.gradW[i,j,k,l] = torch.mul(delta_b_r_c[:,i], input_previous[:,j][:,k:self.out_row*self.stride+k:self.stride,l:l+self.stride*self.out_col:self.stride] ).sum()
			self.gradB[i] = delta_b_r_c[:,i].sum()

		self.gradInput = torch.zeros(input_previous.shape,dtype = torch.double,device=device)
		new_shape = [1,self.W.shape[0],self.W.shape[1],self.W.shape[2], self.W.shape[3]]
		W_new = self.W.reshape(new_shape)
		for orow in range(self.out_row):
			for ocol in range(self.out_col):
				self.gradInput[:,:,orow*self.stride:orow*self.stride+self.f_size,ocol*self.stride:ocol*self.stride+self.f_size] += torch.mul(W_new,delta_b_r_c[:,:,orow,ocol].reshape(n,self.out_depth,1,1,1)).sum(1)
		
		self.W  -= lrate * self.gradW
		self.B  -= lrate * self.gradB

		return self.gradInput

class FlattenLayer:
    def __init__(self):
    	self.output = None
    
    def forward(self, X):
        self.in_batch, self.r, self.c, self.k = X.shape
        self.output = X.reshape(self.in_batch, self.r * self.c * self.k) 
        return self.output

    def backward(self, activation_prev, delta,alpha=None):
        return delta.reshape(self.in_batch, self.r, self.c, self.k)
