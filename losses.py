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


class MSE(Loss):
       def __init__(self):
              self.name = "MSE"
       
       def calculate(self, Y, Y_hat):
              return np.power(Y_hat-Y, 2)
       
       def gradient(self, Y, Y_hat):
              return 2*(Y_hat - Y)
       


class BinaryCrossEntropy(Loss):
       def __init__(self):
              self.name = "hMSE"
       
       def calculate(self, Y, Y_hat):
              return 
       
       def gradient(self, Y, Y_hat):
              """ grad with respect to y_hat
              """
              return 
       


def main():
       return 0

if __name__ == "__main__":
       main()

       



       

    