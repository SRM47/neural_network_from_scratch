from activations import *
from typing import Optional
import numpy as np

class Linear():
       def __init__(self, input_num: int, output_num: int, activation: Activation):
              """
              input: number of nodes in previous layer
              output: number of nodes in current layer
              """
              # first row of the weights matrix is all weights going to the first node in the next layer
              self.W = np.random.normal(0, 2/np.sqrt(input_num), size=(output_num, input_num)) # Kaiming
              self.b = np.random.normal(0, 2/np.sqrt(input_num), size=(output_num))
              self.Z: Optional[np.ndarray] = None # type | type
              self.A_prev: Optional[np.ndarray] = None
              self.A: Optional[np.ndarray] = None
              self.input_num = input_num
              self.output_num = output_num
              self.activation: Activation = activation


       def __call__(self, A: np.ndarray) -> np.ndarray:
              self.A_prev = A
              Z = A @ self.W.T + self.b.T
              self.Z = Z
              self.A = self.activation(Z)
              # print(self.A_prev, self.A)
              return self.A

       def __repr__(self):
              return f"{self.input_num}"
       

def main():
       return 0

if __name__ == "__main__":
       main()