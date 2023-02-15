from abc import ABC, abstractmethod
import numpy as np

class Loss(ABC):
       @abstractmethod
       def calculate(self, Y, Y_hat):
              return
       
       @abstractmethod
       def gradient(self, Y, Y_hat):
              return

       def __call__(self, Y, Y_hat):
              return self.calculate(Y, Y_hat)
       

class hMSE(Loss):
       def __init__(self):
              self.name = "hMSE"
       
       def calculate(self, Y, Y_hat):
              return 0.5*np.power(Y_hat-Y, 2)
       
       def gradient(self, Y, Y_hat):
              """ grad with respect to y_hat
              """
              return Y_hat - Y
       



       

    