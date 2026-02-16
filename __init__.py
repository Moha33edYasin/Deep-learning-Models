__all__ = ("models", "activitions", "data_handler")

from models import * 
from activations import * 
from data_handler import *

def convolve(array, kernel):
    resulted = [[]]
    for y in range(array.shape[0] - kernel.shape[0] - 1):
        end_y = y + kernel.shape[0]
        for x in range(array.shape[1] - kernel.shape[1] - 1):
            end_x = x + kernel.shape[1]

            A = array[y : end_y, x : end_x] * kernel[y : end_y, x : end_x]
            resulted[-1].append(sum(A))
        resulted.append([])

    return np.array(resulted, dtype=float)