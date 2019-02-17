import numpy as np
from activations import *
from layers import *
import gzip
import pickle
from matplotlib import pyplot as plt

class Neural_Network(object):
    
	def __init__(self, hidden_dims=(100,100),n_hidden=2,datapath='./'):

		with gzip.open('mnist.pkl.gz', 'rb') as f:
			self.train_set, self.valid_set, self.test_set = pickle.load(f)
		self.train_images, self.train_labels = self.train_set
		self.valid_images, self.valid_labels = self.valid_set
		self.test_images, self.test_labels = self.test_set

		self.layers = []
		self.layers.append(fc(784,hidden_dims[0]))
		self.layers.append(Sigmoid())
		for i in range (0,n_hidden-1):
			self.layers.append(fc(hidden_dims[i], hidden_dims[i+1]))	
			self.layers.append(Sigmoid())
		self.layers.append(fc(hidden_dims[n_hidden-1], 10))
		self.layers.append(Softmax())

	def initialize_weights(self,n_hidden,init_method):
		for layer in (self.layers[::2]):
			layer.initialize(init_method)

	def forward(self, input_):
		# Store the inputs for use during backpropagation
		self.inputs = []
		for layer in self.layers:
			self.inputs.append(input_)
			output = layer.forward(input_)
			input_ = output
		return output

        
	def batch_loss(self,cross_entropy_act,batch_size,batch_index):

		target_cross_entropy_act = cross_entropy_act[np.arange(cross_entropy_act.shape[0]), 
							self.train_labels[batch_size*batch_index: batch_size*(batch_index+1)]]
		return -np.log(target_cross_entropy_act).mean()
	def cost(self, cross_entropy_act, targets):
		target_cross_entropy_act = cross_entropy_act[np.arange(cross_entropy_act.shape[0]), targets]
		return -np.log(target_cross_entropy_act).mean()

	def backward(self, input_, grad_wrt_output):
	# Store gradients for parameter updates
		self.grads_wrt_output = []
		for input_, layer in zip(self.inputs[::-1], self.layers[::-1]):
			self.grads_wrt_output.append(grad_wrt_output)
			grad_wrt_input = layer.backward(input_, grad_wrt_output)
			grad_wrt_output = grad_wrt_input
		self.grads_wrt_output = self.grads_wrt_output[::-1]
		return grad_wrt_input

	def update_params(self, input_, grad_wrt_output, learning_rate):
		for input_, grad_wrt_output, layer in zip(self.inputs[::2], self.grads_wrt_output[::2], self.layers[::2]): 
			layer.update_params(input_, grad_wrt_output, learning_rate)

	def return_parameter_grads(self, input_,grad_loss): 
		dW = []
		dWi = []
		self.backward(input_,grad_loss)
		for input_, grad_wrt_output, layer in zip(self.inputs[::2], self.grads_wrt_output[::2], self.layers[::2]): 
			dWi,dbi = layer.parameter_grads( input_, grad_wrt_output)
			dW.append(dWi)
		return dW

	def train(self,nbr_epochs, b_size,l_rate):
		num_epochs = nbr_epochs
		batch_size = b_size
		learning_rate = l_rate
		num_batches = int(self.train_set[0].shape[0] / batch_size)

		for epoch in range(1, num_epochs + 1):
			training_loss = 0
			for i in range(num_batches):
				start = batch_size * i
				stop = batch_size * (i + 1)
				X = self.train_images[start:stop]
				y = self.train_labels[start:stop]

				y_hat = self.forward(X)
				training_loss += self.batch_loss(y_hat,batch_size,i)
				grad_loss = np.zeros_like(y_hat)		
				grad_loss[np.arange(grad_loss.shape[0]), y] = -1 / y_hat[np.arange(grad_loss.shape[0]), y] 
				
				self.backward(X, grad_loss)	
				self.update_params(X, grad_loss, learning_rate)

			y_hat = self.forward(self.valid_images)
			decisions = y_hat.argmax(axis=1)
			accuracy = (decisions ==self.valid_labels).mean()
			avg_training_loss = training_loss/num_batches
			print('Epoch {}: Training Loss: {:.3f} Validation Accuracy: {:.3f}'.format(epoch, avg_training_loss, accuracy))
	

	def test(self):
		y_hat = self.forward(self.test_images)
		decisions = y_hat.argmax(axis=1)
		prediction = (decisions ==self.test_labels).mean()
		print('Test accuracy: {:.3f}'.format(prediction))


	def get_numerical_grad(self, single_input_,labels_, i,N):
		# Store the inputs for use during backpropagation
		self.inputs_numerical = []
		eps = 1/float(N)
		num_grad_sample = np.zeros(10)
		for count, layer in enumerate(self.layers[::2]):
			if count ==1: #consider the weights of second layer
				phi,b = layer.params

				for j in range(0,10):  #consider the first 10 elements of weights
					phi[0,j] += eps
					output_p = self.forward(single_input_)
					loss_p = self.cost(output_p,labels_)
								
					phi[0,j] -= eps #remove eps
					phi[0,j] -= eps #consider phi-eps
					output_m = self.forward(single_input_)
					loss_m = self.cost(output_m,labels_)
						
					num_grad_sample[j] =  (loss_p - loss_m)/(2*eps)	
					phi[0,j] += eps #remove eps

		return num_grad_sample

			

	def validate_grad(self):
		X1 = self.train_images[0:1] #select one sample image
		y1 = self.train_labels[0:1]
		max_dif_by_N = []
		N_values = [10,2*10**2,3*10**3,4*10**4,5*10**5]
		for n in N_values:
			print('N = {}'.format(n))

			numerical_grad_sample = self.get_numerical_grad(X1,y1,2,n)
			print('numerical gradient sample = {}'.format(numerical_grad_sample))

			analytical_grad_sample = np.zeros(10)
			y_hat_sample = self.forward(X1)
			grad_loss_sample = np.zeros_like(y_hat_sample)		
			grad_loss_sample[np.arange(grad_loss_sample.shape[0]), y1] = -1 / y_hat_sample[np.arange(grad_loss_sample.shape[0]), y1] 
				
			self.backward(X1, grad_loss_sample)	
			dW = self.return_parameter_grads( X1,grad_loss_sample)
			count =0		
			for dWi in dW:
			
				if count ==1: #consider the weights of second layer
					for j in range(0,10):
						analytical_grad_sample[j] = dWi[0,j]	
				count +=1

			print('analytical gradient sample = {}'.format(analytical_grad_sample))
			
			max_dif = np.amax(np.absolute(numerical_grad_sample - analytical_grad_sample))
			print('max_dif_abs = {}'.format(max_dif))
			max_dif_by_N.append(max_dif)

		print('max_dif_by_N = {}'.format(max_dif_by_N))
		plt.plot(np.log10(N_values),np.log10(max_dif_by_N))
		plt.xlabel('N values (log10 scale)')
		plt.ylabel('Maximum difference (log10 scale)')
		plt.show()



###########################
#    MLP
###########################
num_epochs = 10
batch_size = 100
learning_rate = 0.01
mlp = Neural_Network((100,0),1)
mlp.initialize_weights(1,'glorot') # 'normal' # 'zero' #'glorot'
mlp.train(num_epochs, batch_size,learning_rate)
mlp.test()
mlp.validate_grad()


