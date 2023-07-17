import pandas as pd
import numpy as np
from typing import Optional
from activations import Tanh, ReLU, Base
from layers import Linear
from model import ANN
from optimizers import Adam
from losses import MSELoss, HMSELoss
from train import Trainer
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
       # Create the artificial neural network model.
       n_x = 3
       n_y = 2
       layers = (n_x, 8, 16, 32, 16, 8, n_y)
       activations = (ReLU(), ReLU(), ReLU(),ReLU(),ReLU(),ReLU())
       model = ANN(layers, activations)

       # Synthesize training data.
       N = 100000
       X = np.random.rand(N, n_x)
       Y = create_Y(X)

       # Fit the model to the training data.
       learning_rate = 0.0009
       num_epochs = 10
       trainer = Trainer(
              optimizer = Adam(learning_rate=learning_rate),
              loss_function = MSELoss(),
              num_epochs = num_epochs,
              batch_size = 64)
       losses = trainer.train(
              model = model,
              input = X,
              target = Y,
              method = "minibatch",
              plot_loss = True)
       
       # Plot loss over time.
       g = sns.lineplot(data=losses)
       fig = g.get_figure()
       fig.savefig("loss.png")

       # Create testing data.
       X_test = np.random.rand(5, n_x)
       Y_test = create_Y(X_test)

       # Use the model for inference.
       predictions = model(X_test)
       for y, y_hat in zip(Y_test, predictions):
              print(f"Guess: {y_hat} -> True: {y}")

if __name__ == "__main__":
       main()





