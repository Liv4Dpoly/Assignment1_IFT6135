
"""
Types of layer :

- fully connected

"""
import numpy as np
class fc():
	def __init__(self, in_dim, out_dim):
		self.in_dim = in_dim
		self.out_dim = out_dim
		#default parameter initialization
		W = np.random.randn(self.in_dim, self.out_dim)
		b = np.random.randn( self.out_dim)
		self.params = [W, b]

	def initialize(self,init_method):
		if init_method =='normal':
			W = np.random.randn(self.in_dim, self.out_dim)
			b = np.zeros(self.out_dim)
		elif init_method =='glorot':
			d_low = -np.sqrt(6.0/(self.in_dim + self.out_dim))
			d_high = np.sqrt(6.0/(self.in_dim + self.out_dim))
			W = np.random.uniform(d_low, d_high, (self.in_dim, self.out_dim))
			b = np.zeros(self.out_dim)
		elif init_method =='zero':
			W = np.zeros((self.in_dim, self.out_dim))
			b = np.zeros(self.out_dim)			

		self.params = [W, b]

	def forward(self, input_):
		W, b = self.params
		return input_.dot(W) + b

	def backward(self, input_, grad_wrt_output):
		W, _ = self.params
		return grad_wrt_output.dot(W.T)

	def parameter_grads(self, input_, grad_wrt_output):
		W, b = self.params
		return input_.T.dot(grad_wrt_output), grad_wrt_output.sum(axis=0)

	def update_params(self, input_, grad_wrt_output, learning_rate):
		updates = self.parameter_grads(input_, grad_wrt_output)
		for parameter, update in zip(self.params, updates):
			parameter -= learning_rate * update
