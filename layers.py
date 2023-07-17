from activations import Activation
from typing import Optional
import numpy as np

class Linear():
       def __init__(self, input_num: int, output_num: int, activation: Activation):
              """
              input_num: number of nodes in previous layer
              output_num: number of nodes in current layer
              activation: The activation function for that layer

              Forward pass for an input X:  activation(X*W^T + B)

              The weight matrix, W, contains all the parameters that comprise
              the linear combination of the inputs of a one layer to the next.

              Each row, i, in the weight matrix W, represents the weights that 
              connect the previous layers nodes' output to the ith neuron in the next layer.

              The parameter matrices W and b are initialized with Kaiming He initialization.

              """
              self.W: np.ndarray = np.random.normal(0, 2/np.sqrt(input_num), size=(output_num, input_num))
              self.b: np.ndarray = np.random.normal(0, 2/np.sqrt(input_num), size=(output_num))
              self.W_grad: Optional[np.ndarray] = None
              self.b_grad: Optional[np.ndarray] = None
              self.input_num: int = input_num
              self.output_num: int = output_num
              self.activation: Activation = activation

       def update_gradients(self, weight_gradients: np.ndarray, bias_gradients: np.ndarray):
              self.W_grad = weight_gradients
              self.b_grad = bias_gradients

       def __call__(self, input: np.ndarray) -> Optional[np.ndarray]:
              # Store the output of the previous layer for backpropagation.
              self.A_prev = input
              self.Z = input @ self.W.T + self.b.T
              # Calculate and store the forward pass
              self.A = self.activation(self.Z)
              return self.A

       def __repr__(self) -> str:
              return f"{self.input_num}"