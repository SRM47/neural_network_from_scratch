from optimizers import Optimizer
from model import ANN
from losses import Loss
import numpy as np

class Trainer():
       def __init__(self, optimizer: Optimizer, loss_function: Loss, num_epochs: int, batch_size: int):
              self.optimizer: Optimizer = optimizer
              self.loss_function: Loss = loss_function
              self.epochs: int = num_epochs
              self.batch_size: int = batch_size

       def _shuffled_batch(self, a, b, batch_num):
              assert len(a) == len(b)
              p = np.random.permutation(len(a))
              batch_size = len(p)//batch_num
              for i in range(0,len(p), batch_size):
                     yield a[p][i:i+batch_size, : ], b[p][i: i+batch_size, : ]
       
       def _backward(self, truth: np.ndarray, prediction: np.ndarray):
              """
              Calculates the gradients of the parameters in the neural network
              """
              if not self.model: raise AttributeError("Model is not defined.")
              dldz = self.loss_function.gradient(truth, prediction)
              previous_weight_matrix = np.array([])
              for i, layer in enumerate(self.model.layers[::-1]):
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
       
       def train(self, model: ANN, input: np.ndarray, target: np.ndarray, method: str = "stochastic", plot_loss: bool = False):
              """
              The trainer class is not bound to a specific model. The trainer class represents a specific method of training.
              The train function trains a model `model` on `input` and `target` as X and Y.
              """
              assert method in ["stochastic", "minibatch", "batch"]
              loss_over_time = np.array([])
              self.optimizer.set_model(model)
              self.model = model
              if method == "minibatch":
                     for _ in range(self.epochs):
                            for x_batch, y_batch in self._shuffled_batch(input, target, self.batch_size):
                                   prediction_batch = self.model(x_batch)
                                   loss_over_time = np.append(loss_over_time, [self.loss_function(y_batch, prediction_batch)])
                                   self._backward(y_batch, prediction_batch)
                                   self.optimizer.step()
              elif method == "batch":
                     for _ in range(self.epochs):
                            X, Y = next(self._shuffled_batch(input, target, 1))
                            prediction = self.model(X)
                            loss_over_time = np.append(loss_over_time, self.loss_function(Y, prediction))
                            self._backward(Y, prediction)
                            self.optimizer.step()
              else:
                     for _ in range(self.epochs):
                            for x, y in self._shuffled_batch(input, target, len(input)):
                                   prediction = self.model(x)
                                   loss_over_time = np.append(loss_over_time, [self.loss_function(y, prediction)])
                                   self._backward(y, prediction)
                                   self.optimizer.step()
              self.model = None
              return loss_over_time if plot_loss else None


