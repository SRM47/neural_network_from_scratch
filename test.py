import pandas as pd
import numpy as np
from typing import Optional
from activations import *
from layers import Linear
from model import NN
from losses import hMSE
import seaborn as sns


def create_Y(X):
       # y = 5*x1 + 3*x2 + 3*x3
       Y = 0.5*X[:, 0] + 0.4*X[:, 1] + 0.1*X[:, 2] 
       return Y.reshape((Y.shape[0], 1))


def main():
       n_x, n_y = 3, 1
       nn = NN(layers = (n_x, 2, 4, 2, n_y), 
               activations = (Tanh(), ReLU(), ReLU(), Base()),
               loss = hMSE(),
               lr=0.002,)

       # create training data
       N = 10000 # sample size
       X = np.random.rand(N, n_x)
       Y = create_Y(X)

       # train
       losses = nn.train(X, Y, 1)
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





