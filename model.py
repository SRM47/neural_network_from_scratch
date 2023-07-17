import numpy as np
from typing import Optional
from activations import Activation
from layers import Linear
from losses import Loss, HMSELoss

class ANN():
       def __init__(self, layers: tuple, activations: tuple):
              self.layers: list[Linear] = []
              # Add the layers into the neural network
              for i in range(len(layers)-1):
                     self.add_layer(layers[i], layers[i+1], activations[i])

       def forward(self, input: np.ndarray) -> np.ndarray:
              """
              This function implements the forward pass of neural network.
              input: The optionally batched input to the neural network. 
              The shape of this array is (batch_size, # of input features).
              """
              output = input
              for layer in self.layers:
                     output = layer(output)
              return output

       def add_layer(self, input: int, output: int, activation: Activation):
              self.layers.append(Linear(input, output, activation))

       def __call__(self, input: np.ndarray) -> np.ndarray:
              return self.forward(input)




def main():
       return 0

if __name__ == "__main__":
       main()

       


