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
              self.beta_s = 0.999
              self.beta_m = 0.9
              for i in range(len(layers)-1):
                     self.add_layer(layers[i], layers[i+1], activations[i])


       def backward(self, Y, y_hat, adam_params, counter):
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
                     # dlossdz[l+1] is stored in `prev` because we need in future calculation
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

                     
                     # layer.W -= self.lr*dldw.mean(axis=0)
                     # layer.b -= self.lr*dldb.mean(axis=0)
                     new_weights.append((dldw.mean(axis=0), dldb.mean(axis=0)))
              
              # implement Adam Optimizer
              for i, layer in enumerate(self.layers[::-1]):
                     updates = new_weights[i]
                     g_w = updates[0]
                     g_b = updates[1]
                     momentum_bias_correction = 1/(1-np.power(self.beta_m, counter))
                     squares_bias_correction = 1/(1-np.power(self.beta_s, counter))

                     m_w_prev, s_w_prev, m_b_prev, s_b_prev = adam_params[i][0], adam_params[i][1], adam_params[i][2], adam_params[i][3] 

                     # for the weights
                     m_w = (self.beta_m)*m_w_prev + (1-self.beta_m)*g_w
                     s_w = (self.beta_s)*s_w_prev + (1-self.beta_s)*np.power(g_w, 2)
                     layer.W -= self.lr * (m_w/momentum_bias_correction)/(np.sqrt(s_w/squares_bias_correction) + 1e-8) # updates[0]

                     # for the bias
                     m_b = (self.beta_m)*m_b_prev + (1-self.beta_m)*g_b
                     s_b = (self.beta_s)*s_b_prev + (1-self.beta_s)*np.power(g_b, 2)
                     layer.b -= self.lr * (m_b/momentum_bias_correction)/(np.sqrt(s_b/squares_bias_correction) + 1e-8) # updates[1]

                     adam_params[i] = (m_w, s_w, m_b, s_b)

       def shuffled_batch(self, a, b, batch_num):
              assert len(a) == len(b)
              p = np.random.permutation(len(a))
              batch_size = len(p)//batch_num
              for i in range(0,len(p), batch_size):
                     yield a[p][i:i+batch_size, : ], b[p][i: i+batch_size, : ]
       
       def train(self, X, Y, num_epochs, batch_num = 1000, method = "stochastic"):
              assert method in ["stochastic", "minibatch", "batch"]

              loss_over_time = np.array([])

              if method == "minibatch":
                     adam_params = [(0,0,0,0) for _ in self.layers]
                     for epoch in range(num_epochs):
                            counter = 0
                            for x,y in self.shuffled_batch(X, Y, batch_num):
                                   counter += 1
                                   y_hat = self.forward(x)
                                   loss_over_time = np.append(loss_over_time, [self.loss(y, y_hat)])
                                   self.backward(y, y_hat, adam_params, counter)
              elif method == "batch":
                     adam_params = [(0,0,0,0) for _ in self.layers]
                     for epoch in range(num_epochs):
                            Y_hat = self.forward(X)
                            loss_over_time.append(self.loss(Y, Y_hat))
                            self.backward(Y, Y_hat, adam_params, epoch + 1)
              else:
                     # loss carries over through epochs as well. 
                     adam_params = [(0,0,0,0) for _ in self.layers]
                     for epoch in range(num_epochs):
                            counter = 0
                            for x,y in zip(X,Y):
                                   counter += 1
                                   x = x.reshape(1, x.shape[0]) # must be row vector
                                   y = y.reshape(1, y.shape[0]) # must be row vector as well
                                   y_hat = self.forward(x)
                                   loss_over_time = np.append(loss_over_time, [self.loss(y, y_hat)])
                                   self.backward(y, y_hat, adam_params, counter)


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

       


