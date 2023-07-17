from abc import ABC, abstractmethod
from typing import Optional
import numpy as np
from model import ANN

class Optimizer(ABC):
       """
       The Optimizer class is an abstract class that scaffolds an optimizer.
       """
       def __init__(self, model: Optional[ANN] = None, learning_rate: float = 0.1):
              self.model = model
              self.lr: float = learning_rate

       @abstractmethod
       def step(self):
              raise NotImplementedError(
                     "This optimizer hasn't been implemented yet")

       @abstractmethod
       def zero_grad(self) -> bool:
              raise NotImplementedError(
                     "This optimizer hasn't been implemented yet")
       
       @abstractmethod
       def reset(self):
              raise NotImplementedError(
                     "This optimizer hasn't been implemented yet")

       def set_model(self, model: ANN):
              self.model = model
              self.reset()
       
class Adam(Optimizer):
       def __init__(self, model: Optional[ANN] = None, learning_rate: float = 0.1, beta_s: float = 0.999, beta_m: float = 0.9):
              super().__init__(model, learning_rate)
              self.params: Optional[list[tuple]] = [(0,0,0,0) for _ in self.model.layers] if self.model else None
              self.beta_s: float = beta_s
              self.beta_m: float = beta_m
              # Private variable used for 
              self._counter: int = 1

       def zero_grad(self) -> bool:
              if not self.model: return False
              params = []
              for layer in self.model.layers[::-1]:
                     layer.W_grad = None
                     layer.b_grad = None
                     params.append((0,0,0,0))
              self.params = params
              return True
       
       def reset(self):
              if self.zero_grad():
                     self.counter = 1

       def step(self):
              if not self.model or not self.params:
                     raise AttributeError("Optimizer must have a model. Use `set_model` to do so.")
              
              for i, layer in enumerate(self.model.layers[::-1]):
                     g_w, g_b = layer.W_grad, layer.b_grad
                     if (g_w is None or g_b is None): continue

                     momentum_bias_correction = 1/(1-np.power(self.beta_m, self._counter))
                     squares_bias_correction = 1/(1-np.power(self.beta_s, self._counter))

                     m_w_prev, s_w_prev, m_b_prev, s_b_prev = self.params[i][0], self.params[i][1], self.params[i][2], self.params[i][3] 

                     # Update the weight matrix
                     m_w = (self.beta_m)*m_w_prev + (1-self.beta_m)*g_w
                     s_w = (self.beta_s)*s_w_prev + (1-self.beta_s)*np.power(g_w, 2)
                     layer.W -= self.lr * (m_w/momentum_bias_correction)/(np.sqrt(s_w/squares_bias_correction) + 1e-8) # updates[0]

                     # Update the bias matrix
                     m_b = (self.beta_m)*m_b_prev + (1-self.beta_m)*g_b
                     s_b = (self.beta_s)*s_b_prev + (1-self.beta_s)*np.power(g_b, 2)
                     layer.b -= self.lr * (m_b/momentum_bias_correction)/(np.sqrt(s_b/squares_bias_correction) + 1e-8) # updates[1]

                     self.params[i] = (m_w, s_w, m_b, s_b)
              self._counter += 1
