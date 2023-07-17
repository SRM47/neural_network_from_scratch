import numpy as np
from typing import Any
from abc import ABC, abstractmethod

class Activation(ABC):
       """
       This Activation abstract class scaffolds a layer's activation function
       """
       def __init__(self, name: str):
              self.name = "Activation abstract bass class"
              
       @abstractmethod
       def __call__(self, input) -> np.ndarray:
              return 
       
       @abstractmethod
       def gradient(self, input) -> np.ndarray:
              return 
       
       def __repr__(self) -> str:
              return self.name
       
class Sigmoid(Activation):
       def __init__(self):
              super().__init__("Sigmoid")
       
       def __call__(self, input: np.ndarray) -> np.ndarray:
              return 1/(1 + np.exp(-input))

       def gradient(self, input: np.ndarray) -> np.ndarray:
              return self(input)*(1-self(input))
       

class Tanh(Activation):
       def __init__(self):
              super().__init__("Hyperbolic Tangent")
       
       def __call__(self, input: np.ndarray) -> np.ndarray:
              return np.tanh(input)

       def gradient(self, input: np.ndarray) -> np.ndarray:
              return 1-np.power(np.tanh(input), 2)

class ReLU(Activation):
       def __init__(self):
              super().__init__("Rectified Linear Unit")
       
       def __call__(self, input: np.ndarray) -> np.ndarray:
              return np.maximum(0, input)

       def gradient(self, input: np.ndarray) -> np.ndarray:
              input[input<=0] = 0
              input[input>0] = 1
              return input

class Base(Activation):
       def __init__(self):
              super().__init__("Identity")
       
       def __call__(self, input: np.ndarray) -> np.ndarray:
              return input
       
       def gradient(self, input: np.ndarray) -> float:
              return 1.0


def main():
       return 0

if __name__ == "__main__":
       main()

