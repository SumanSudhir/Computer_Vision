import Model
from Linear import LinearLayer 
from ReLu import ReLu
from Criterion import Criterion
from Convolution import ConvolutionLayer, FlattenLayer

import torchfile
import torch
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# device = torch.device('gpu')
print(device)
def train(model_one,Train_Data,Train_Label,alpha=3.0e-5,epochs=50):
	# model_one = Model.Model()

	# # model_one.addLayer(ConvolutionLayer( (1,108,108) , 12, 12, 8))
	# # model_one.addLayer(ReLu())
	# # model_one.addLayer(ConvolutionLayer( (12,13,13) , 4 , 9, 3)) #9,4,4
	# # model_one.addLayer(FlattenLayer())
	# # model_one.addLayer(ReLu())
	# # model_one.addLayer(LinearLayer(144,36))
	# # model_one.addLayer(ReLu())
	# # model_one.addLayer(LinearLayer(36,12))
	# # model_one.addLayer(ReLu())
	# # model_one.addLayer(LinearLayer(12,6))

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

	# model_one = torch.load("Model_3e-05_70")

	# Train_Data = torchfile.load('data.bin')
	# Train_Label = torchfile.load('labels.bin')
	# print(Train_Data.shape)

	# No_of_points,No_of_points_validation = 20,10
	# No_of_points,No_of_points_validation = 80,20
	# No_of_points,No_of_points_validation = 200,40
	# No_of_points,No_of_points_validation = 400,100
	# No_of_points,No_of_points_validation = 1000,200
	# # # No_of_points,No_of_points_validation = 2000,400
	# # No_of_points,No_of_points_validation = 5000,1000
	# # No_of_points,No_of_points_validation = 10000,2000
	# No_of_points,No_of_points_validation = 20000,4000
	No_of_points,No_of_points_validation = 2510,4000
	# No_of_points,No_of_points_validation = Train_Data.shape[0],20
	# No_of_points = 8000

	Start_validation = No_of_points
	# No_of_points_validation = 1000
	# No_of_points_validation = 2000

	print("Train:Val",No_of_points,No_of_points_validation)


	Val_Data = torch.from_numpy(Train_Data  [ Start_validation:Start_validation +No_of_points_validation]).double().to(device)
	Val_Label = torch.from_numpy(Train_Label[ Start_validation:Start_validation +No_of_points_validation]).long().to(device)
	Val_Data = Val_Data.reshape( No_of_points_validation ,1, 108,108)


	Train_Data = torch.from_numpy(Train_Data[0:No_of_points]).double().to(device)
	Train_Label = torch.from_numpy(Train_Label[0:No_of_points]).long().to(device)
	Train_Data = Train_Data.reshape(No_of_points,1,108,108)
	# Train_Data = Train_Data[0:No_of_points]/255.0
	# Train_Data = Train_Data -  Train_Data.mean(1).reshape(No_of_points,1)

	# print("percentage",percentage)


	No_of_points_10 = int(No_of_points/10)
	batch_size = 50
	No_of_batch = int(No_of_points/batch_size)
	No_of_points_10 = batch_size

	# epochs = 50
	# alpha =2.8e-5
	print("epochs:",epochs,"   alpha:",alpha)
	print("Val-accuracy")
	for i in range(epochs):

		if i%5 == 0:
		    f_l_w_p = model_one.forward(Val_Data)
		    _,indices = f_l_w_p.max(1) 
		    matches = indices == Val_Label
		    percentage = ( 100.0 * matches.sum() )/No_of_points_validation
		    print(float(percentage),"%",end=' ')
		    # print("Model Saving",end=" ... ")
		    # torch.save(model_one,"Model_30+_lr_i10_"+str(alpha)+"_"+ str(i))
		    # print("Model Saved", end= "\t")

	    # print("ith '%'loss ",i,float(100*percentage),"%\t\t", Criterion.forward(f_l_w_p,Train_Label))
		
		for j in range(No_of_batch):
			final_label_without_prob =  model_one.forward(Train_Data[j* No_of_points_10:(j+1)* No_of_points_10])
	        # print(final_label_without_prob[0:10])
			backward_gradient = Criterion.backward(final_label_without_prob,Train_Label[j* No_of_points_10:(j+1)* No_of_points_10])
			model_one.backward(Train_Data[j* No_of_points_10:(j+1)* No_of_points_10],backward_gradient,alpha)
			# model_one.backward(Train_Data[j* No_of_points_10:(j+1)* No_of_points_10],backward_gradient,1.0e-5/No_of_points)

		final_label_without_prob =  model_one.forward(Train_Data)
		_,indices = final_label_without_prob.max(1) 
		matches = indices == Train_Label
		# matches = indices.long() == Train_Label
		percentage = ( 100.0 * matches.sum() )/No_of_points
		print("		",i,"ith '%':loss ",float(percentage),"%\t:\t", Criterion.forward(final_label_without_prob,Train_Label))
		# backward_gradient = Criterion.backward(final_label_without_prob,Train_Label)
		# model_one.backward(Train_Data,backward_gradient)

	    # print("my label",final_label_without_prob[0:10])
	    # print("backward gradient, ", backward_gradient[0:10])
	    # print("real label",Train_Label[0:10])
	    # model_one.backward(Train_Data,backward_gradient)
	    # print(i,"th iteration completed")
	    # _,indices = model_one.forward(Train_Data).max(1) 
	    # print(indices)


	# print("Model Saving",end=" ... ")
	# torch.save(model_one,"Model_30+_"+str(alpha)+"_"+str(epochs))
	# print("Model Saved", end= "\t")

	_,indices = model_one.forward(Train_Data).max(1) 
	matches = indices == Train_Label
	accuracy = matches.sum()
	percentage = accuracy.double()/No_of_points
	print("\n\n\n\n")
	print("percentage",float(percentage)*100,"%")
	return model_one