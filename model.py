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
              prev = None
              prev_weight = None
              new_weights = []
              # loop through layers from back to front
              for i, layer in enumerate(self.layers[::-1]):
                     # i=0 is the condition where we're focusing on the last layer
                     # here, we're trying to calculate dldz because dldz is present
                     # in the calculation of dldw and dldb for that layer, and we also need dldz of 
                     # the current layer to calculate the dldz of the previous (next) layer

                     # if its the last layer, dlossdz = dlossdy * dydz (dydz is the same as dAdz)
                     # if we're at an intermediate layer, dlossdz[l] =  dlossdz[l+1] * dz[l+1]dA[l] * dA[l]dz[l]
                     # dlossdz[l+1] is stored in `prev`` because we need in future calculation
                     # dz[l+1]dA[l] is from something like Z3 = A2@W3T + B3T, so the result is W3T, which is the previous (technically next) layer's weight matrix
                     # dA[l]dz[l] is from A = f(Z) where f is the activation function
                     # A = f(Z), Z = A[l-1]W.T + B.T
                     dldz = self.loss.gradient(Y, y_hat) if i == 0 else (prev @ prev_weight)
                     dldz *= layer.activation.gradient(layer.Z)

                     # store dldz for the current layer to use in the next iteration
                     prev = dldz

                     # dldw = dldz * dzdw
                     dldw = dldz.T @ layer.A_prev
                     # i thought it was dldw = dldz @ layer.A_prev at first
                     dldb = dldz.mean(axis=0) # why the mean dim/axis 0?

                     # store current weight matrix for next iterations calculations
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

       


