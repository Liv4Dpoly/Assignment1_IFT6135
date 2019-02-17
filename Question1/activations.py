"""
Types of activations :

- Sigmoid
- Softmax

"""
import numpy as np


class Sigmoid:
	def forward(self, input_):
		self.output = 1 / (1 + np.exp(-input_))
		return self.output
	def backward(self, input_, grad_wrt_output):
		return grad_wrt_output * self.output * (1 - self.output)

class Relu:
	def forward(self, input_):
		self.output = np.maximum(input_, 0)
		return self.output 
	def backward(self, input_, grad_wrt_output):
		return (grad_wrt_output * (self.output > 0))

class Softmax:
	def forward(self, input_):
		exp_input = np.exp(input_ - input_.max(axis=1, keepdims=True))
		self.output = exp_input / exp_input.sum(axis=1, keepdims=True)
		return self.output
	def backward(self, input_, grad_wrt_output):
		gx = self.output * grad_wrt_output
		gx -= self.output * gx.sum(axis=1, keepdims=True)
		return gx


