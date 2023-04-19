import pandas as pd
import numpy as np
from typing import Optional
from activations import *
from layers import Linear
from model import NN
from losses import *
import seaborn as sns
import scipy 


def create_Y(X):

       Y_0 = 3*X[:, 0] - 0.4*np.power(X[:, 1],2) * 0.1*X[:, 2]
       Y_0 = Y_0.reshape((Y_0.shape[0], 1))
       Y_1 = -0.6*np.power(X[:, 0],2) + 7*np.power(X[:, 2],2)
       Y_1 = Y_1.reshape((Y_1.shape[0], 1))
       d = np.concatenate([Y_0, Y_1], axis=1)
       return  d + np.random.normal(0,1, size=d.shape)


def main():
       n_x, n_y = 3, 2
       nn = NN(layers = (n_x, 16, 32, 16, n_y), 
               activations = (Tanh(), ReLU(), Tanh(), Base()),
               loss = hMSE(),
               lr=0.001,)

       # create training data
       N = 10000 # sample size
       X = np.random.rand(N, n_x)
       Y = create_Y(X)
       # print(Y)

       # train
       losses = nn.train(X, Y, 10, batch_num = 100, method="minibatch")
       g = sns.lineplot(data=losses)
       fig = g.get_figure()
       fig.savefig("loss.png")

       # create testing data
       X_test = np.random.rand(5,n_x)
       Y_test = create_Y(X_test)
       Y_hat = nn.predict(X_test)
       # my predictions are all very huge! idk why
       for y, y_hat in zip(Y_test, Y_hat):
              print(f"Guess: {y_hat} -> True: {y}")


if __name__ == "__main__":
       main()





