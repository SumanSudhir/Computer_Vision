import torch
# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
device = torch.device('cpu')


class Model:

    def __init__(self):
        self.layers = []
        self.isTrain = False #True when training
    
    def forward(self,input):
        output = input.clone().to(device)
        for layer in self.layers:
            output = layer.forward(output)
        return output

    def backward(self,input, gradOutput,alpha=None):



        ##why
        # self.clearGradParam()
        # self.clearGradParam()
        # print("gradoutput\t",gradOutput)
        # gradOutput1 = gradOutput
        for i in range(len(self.layers)-1, 0 , -1):
            gradOutput = self.layers[i].backward(  self.layers[i-1].output , gradOutput ,alpha)
        self.layers[0].backward(input  ,  gradOutput , alpha)

    def dispGradParam(self):
        #parameters 
        for i in range(len(self.layers)-1, -1 , -1):
            print("for layer no: ", i , self.layers[i].gradInput)

    def clearGradParam(self):
        for layer in self.layers:
            layer.gradInput = None
            # layer.gradInput.zero_grad()
            # layer.gradInput = torch.zeros(layer.gradInput.shape,dtype = torch.double)

    #addLayer(Layer class object) is used to add an object of type Layer to the Layers table.
    def addLayer(self,class_object):
        self.layers.append(class_object)