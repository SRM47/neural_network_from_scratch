import numpy as np
from typing import Optional
from activations import Activation
from layers import Linear
from losses import Loss, HMSELoss
from optimizers import Optimizer, Adam


class NN():
       def __init__(self, layers: tuple, activations: tuple, loss_function: Loss):
              self.layers = []
              # Add the layers into the neural network
              for i in range(len(layers)-1):
                     self.add_layer(layers[i], layers[i+1], activations[i])
              self.loss = loss_function


       def backward(self, truth: np.ndarray, prediction: np.ndarray):
              """
              Calculates the gradients of the parameters in the neural network
              """
              dldz = self.loss.gradient(truth, prediction)
              previous_weight_matrix = None
              for i, layer in enumerate(self.layers[::-1]):
                     # Calculate ∂Loss/∂A for the current layer.
                     dlda = dldz if i == 0 else dldz @ previous_weight_matrix
                     # Store current layer's weight matrix for next iterations calculations.
                     previous_weight_matrix = layer.W
                     # Calculate ∂Loss/∂Z for the current layer; It will be used in the next iteration.
                     dldz = dlda * layer.activation.gradient(layer.Z)
                     # Calculate ∂l/∂w and ∂l/∂b for the current layer.
                     dldw = (dldz.T @ layer.A_prev).mean(axis=0)
                     dldb = dldz.mean(axis=0)
                     # Average the gradients across the rows (batches) and update the layer's gradients
                     layer.update_gradients(dldw, dldb)

       def shuffled_batch(self, a, b, batch_num):
              assert len(a) == len(b)
              p = np.random.permutation(len(a))
              batch_size = len(p)//batch_num
              for i in range(0,len(p), batch_size):
                     yield a[p][i:i+batch_size, : ], b[p][i: i+batch_size, : ]
       
       def train(self, input: np.ndarray, target: np.ndarray, num_epochs: int, optimizer: Optimizer, batch_num: int = 1000, method: str = "stochastic", plot_loss: bool = False):
              assert method in ["stochastic", "minibatch", "batch"]

              loss_over_time = np.array([])

              if method == "minibatch":
                     for _ in range(num_epochs):
                            for x_batch, y_batch in self.shuffled_batch(input, target, batch_num):
                                   prediction_batch = self(x_batch)
                                   loss_over_time = np.append(loss_over_time, [self.loss(y_batch, prediction_batch)])
                                   self.backward(y_batch, prediction_batch)
                                   optimizer.step()
              elif method == "batch":
                     for _ in range(num_epochs):
                            prediction = self(input)
                            loss_over_time = np.append(loss_over_time, self.loss(target, prediction))
                            self.backward(target, prediction)
                            optimizer.step()
              else:
                     for _ in range(num_epochs):
                            for x, y in self.shuffled_batch(input, target, 1):
                                   # Reshape single training instances to row vectors.
                                   x = x.reshape(1, x.shape[0])
                                   y = y.reshape(1, y.shape[0])
                                   prediction = self(x)
                                   loss_over_time = np.append(loss_over_time, [self.loss(y, prediction)])
                                   self.backward(y, prediction)
                                   optimizer.step()


              return loss_over_time

       def forward(self, input: np.ndarray) -> np.ndarray:
              """
              This function implements the forward pass of neural network.
              input: The optionally batched input to the neural network. 
              The shape of this array is (batch_size, # of input features).
              """
              res = input
              for layer in self.layers:
                     res = layer(res)
              return res

       def add_layer(self, input: int, output: int, activation: Activation):
              self.layers.append(Linear(input, output, activation))

       def __call__(self, input: np.ndarray) -> np.ndarray:
              return self.forward(input)




def main():
       return 0

if __name__ == "__main__":
       main()

       


