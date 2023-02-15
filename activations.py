import numpy as np

class Activation():
       def __init__(self):
              self.temp = 0
              

       def __call__(self, data):
              return data
       
       def gradient(self, data):
              return 1
       
class Sigmoid(Activation):
       def __init__(self):
              super().__init__()
       
       def __call__(self, data):
              return 1/(1 + np.exp(-data))


       def gradient(self, data):
              return self(data)*(1-self(data))
       

class Tanh(Activation):
       def __init__(self):
              super().__init__()
       
       def __call__(self, data):
              return np.tanh(data)


       def gradient(self, data):
              return 1-np.power(np.tanh(data), 2)

class ReLU(Activation):
       def __init__(self):
              super().__init__()
       
       def __call__(self, data):
              return np.maximum(0, data)

       def gradient(self, data):
              data[data<=0] = 0
              data[data>0] = 1
              return data

class Base(Activation):
       def __init__(self):
              super().__init__()


def main():
       return 0

if __name__ == "__main__":
       main()

