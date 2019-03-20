import torch
import numpy as np
inputs = [43, 69, 58, 4, 66, 68, 73, 69, 62, 69, 60, 13, 4, 76, 63, 58, 69, 4, 35, 73, 58, 60, 68, 73, 4]
hidden_size = 100
output_size = 1
lr = 1e-1

class RNN():
    #model parameters
    Wxh = torch.randn(hidden_size, output_size)*0.01         #Input to hidden
    Whh = torch.randn(hidden_size, hidden_size)*0.01         #Hidden to itself
    Why = torch.randn(output_size, hidden_size)*0.01         #Hidden to output
    Bh =  torch.zeros(hidden_size, 1)                        #Hidden Bias
    By =  torch.zeros(output_size, 1)                        #Output Bias


    def __init__(self, input_size, hidden_size, output_size):
        super(RNN, self).__init__()
        self.hidden_size = hidden_size
        self.xs = {}                                       #will store encoded input
        self.hs = {}                                       #Store hidden state output
        self.ys = {}                                       #Store output
        self.ps = {}                                       #Probabilities of next character
        self.dWxh = torch.zeros_like(Wxh)
        self.dWhh = torch.zeros_like(Whh)
        self.dWhy = torch.zeros_like(Why)
        self.dBh  = torch.zeros_like(Bh)
        self.dBy  = torch.zeros_like(By)

    def forward(self, inputs, h_initial):             #h_initial is Hx1 array of initial hidden state

        hs[-1] = torch.copy(h_initial)
        for i in range(len(inputs)):
            xs[i] = torch.zeros(output_size,1)
            xs[i][inputs[i]] = 1
            #hidden_state = input*Wxh + last_value_of_hidden_layer*Whh + Bias
            hs[i] = torch.tanh(torch.dot(Wxh, xs[i]) + torch.dot(Whh, hs[i-1]) + Bh)        #Hidden State
            ys[i] = torch.dot(Why, hs[i]) + By
            ps[i] = torch.exp(ys[i])/torch.sum(torch.exp(ys[i]))    #Noramlized Probability

            return xs, hs, ys, ps

    def backward(self, inputs):
        dhnext = np.zeros_like(hs[0])
        for i in reversed(range(len(inputs))):
            dy = torch.copy(ps[i])         #Our first gradient
            dy[targets[t]] -= 1             #back prop in y
            dWhy += torch.dot(dy, torch.transpose(hs[t],0,1))
            dBy += dy
            dh = torch.dot(torch.transpose(Why,0,1), dy) + dhnext
            dhraw = (1 - hs[i]*hs[i])*dh
            dBh += dhraw
            dWxh += torch.dot(dhraw, torch.transpose(xs[i],0,1))
            dWhh += torch.dot(dhraw, torch.transpose(hs[i-1],0,1))
            dhnext = torch.dot(torch.transpose(Whh,0,1), dhraw )

        return ys
