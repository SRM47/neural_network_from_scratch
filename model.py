import pandas as pd
import numpy as np
from typing import Optional
from activations import *
from layers import Linear
from losses import Loss, hMSE


class NN():
       def __init__(self, layers: tuple, activations: tuple, loss:Loss, lr=0.1):
              self.layers = []
              self.lr = lr
              self.loss = loss
              for i in range(len(layers)-1):
                     self.add_layer(layers[i], layers[i+1], activations[i])


       def backward(self, Y, y_hat):
              prev = 1
              prev_weight = None
              new_weights = []
              for i, layer in enumerate(self.layers[::-1]):
                     dldz = self.loss.gradient(Y, y_hat) if i == 0 else (prev @ prev_weight)
                     dldz *= layer.activation.gradient(layer.Z)

                     prev = dldz
                     dldw = dldz.T @ layer.A_prev
                     # i thought it was dldw = dldz @ layer.A_prev at first
                     dldb = dldz.mean(axis=0) # why the mean dim/axis 0?
                     prev_weight = layer.W

                     layer.W -= self.lr*dldw
                     layer.b -= self.lr*dldb
                     new_weights.append((self.lr*dldw, self.lr*dldb))
              
              for i, layer in enumerate(self.layers[::-1]):
                     updates = new_weights[i]
                     layer.W -= updates[0]
                     layer.b -= updates[1]
              
       def train(self, X, Y, num_epochs):

              loss_over_time = np.array([])

              # # batch gradient descent
              # for epoch in range(num_epochs):
              #        Y_hat = self.forward(X)
              #        loss_over_time.append(self.loss(Y, Y_hat))
              #        self.backward(Y, Y_hat)

              # stochastic gradient descent
              for epoch in range(num_epochs):
                     for x,y in zip(X,Y):
                            x = x.reshape(1, x.shape[0]) # must be row vector
                            y = y.reshape(1, y.shape[0]) # must be row vector as well
                            y_hat = self.forward(x)
                            loss_over_time = np.append(loss_over_time, [self.loss(y, y_hat)])
                            self.backward(y, y_hat)
              return loss_over_time

       def predict(self, X):
              return self.forward(X)

       def forward(self, data):
              """
              shape of data is (N, n_0)
              """
              res = data
              for layer in self.layers:
                     res = layer(res)
              return res

       def add_layer(self, input, output, activation):
              self.layers.append(Linear(input, output, activation))




def main():
       return 0

if __name__ == "__main__":
       main()

       


