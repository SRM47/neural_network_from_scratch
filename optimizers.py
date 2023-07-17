from abc import ABC, abstractmethod
from typing import Optional
import numpy as np

class Optimizer(ABC):
       """
       The Optimizer class is an abstract class that scaffolds an optimizer.
       """
       def __init__(self, model, learning_rate: float = 0.1):
              self.model = model
              self.lr: float = learning_rate

       @abstractmethod
       def step(self):
              raise NotImplementedError(
                     "This optimizer hasn't been implemented yet")
       
class Adam(Optimizer):
       def __init__(self, model, learning_rate: float = 0.1):
              super().__init__(model, learning_rate)
              self.params: list[tuple] = [(0,0,0,0) for _ in self.model.layers]
              self.beta_s: float = 0.999
              self.beta_m: float = 0.9
              self.counter: int = 1

       def step(self):
              for i, layer in enumerate(self.model.layers[::-1]):
                     g_w, g_b = layer.W_grad, layer.b_grad

                     momentum_bias_correction = 1/(1-np.power(self.beta_m, self.counter))
                     squares_bias_correction = 1/(1-np.power(self.beta_s, self.counter))

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
              self.counter += 1
