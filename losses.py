from abc import ABC, abstractmethod
from typing import Optional, Union, Any
import numpy as np

class Loss(ABC):
       """
       The Loss class is an abstract class that scaffolds a loss function.

       truth and prediction have the same shape (batch_size, num_output_features).
       num_output_features is the number of neurons on the last layer of the network.
       """

       def __init__(self, reduction_function: str = "mean"):
              assert reduction_function in ["mean", "sum"], \
                     "The reduction function must be either `mean`, `sum`"
              self.name: str = "Loss function abstract base class"
              self.reduction: str = reduction_function

       @abstractmethod
       def calculate(self, truth: np.ndarray, prediction: np.ndarray):
              raise NotImplementedError(
                     "The loss function is not yet implemented. \
                     Use one of the implemented loss functions.")
       
       @abstractmethod
       def gradient(self, truth: np.ndarray, prediction: np.ndarray) -> np.ndarray:
              """
              The gradient method implements the partial derivative:

              ∂Loss(truth, prediction)
              ------------------------
                     ∂prediction
              """
              raise NotImplementedError(
                     "The gradient method of this loss function has not yet been implemented. \
                     Use one of the implemented loss functions.")

       def __call__(self, truth: np.ndarray, prediction: np.ndarray):
              return self.calculate(truth, prediction)
       
       def __repr__(self) -> str:
              return self.name
       
       

class HMSELoss(Loss):
       def __init__(self, reduction_function: str = "mean"):
              super().__init__(reduction_function)
              self.name: str = "Half Mean Squared Error Loss"
       
       def calculate(self, truth: np.ndarray, prediction: np.ndarray):
              loss = 0.5 * np.power( prediction - truth , 2)
              return np.sum(loss) if self.reduction == "sum" else np.mean(loss)
       
       def gradient(self, truth: np.ndarray, prediction: np.ndarray) -> np.ndarray:
              loss = (prediction - truth) / (1 if self.reduction == "sum" else prediction.shape[1])
              return loss

class MSELoss(Loss):
       def __init__(self, reduction_function: str = "mean"):
              super().__init__(reduction_function)
              self.name: str = "Mean Squared Error Loss"
       
       def calculate(self, truth: np.ndarray, prediction: np.ndarray):
              loss = np.power( prediction - truth , 2)
              return np.sum(loss) if self.reduction == "sum" else np.mean(loss)
       
       def gradient(self, truth: np.ndarray, prediction: np.ndarray) -> np.ndarray:
              loss = 2 * (prediction - truth) / (1 if self.reduction == "sum" else prediction.shape[1])
              return loss
       

class CrossEntropy(Loss):
       def __init__(self, reduction_function: str = "mean"):
              super().__init__(reduction_function)
              self.name: str = "Multiclass Cross Entropy Loss"

       



       

    