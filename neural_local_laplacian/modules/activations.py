# Third-party imports
import torch
from torch import Tensor


class Sine(torch.nn.Module):
    """
    A custom activation function implementing the sine function.

    This module applies the sine function element-wise to the input tensor.
    It can be used as a drop-in replacement for other activation functions
    in neural network architectures.
    """

    def __init__(self) -> None:
        """
        Initialize the Sine activation function.
        """
        super().__init__()

    def forward(self, input: Tensor) -> Tensor:
        """
        Apply the sine function to the input tensor.

        Args:
            input (Tensor): The input tensor.

        Returns:
            Tensor: The output tensor after applying the sine function.
        """
        return torch.sin(input)

    def __repr__(self) -> str:
        """
        Return a string representation of the Sine activation function.

        Returns:
            str: A string representation of the object.
        """
        return f"{self.__class__.__name__}()"