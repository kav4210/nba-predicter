"""Exercise 4: temperature scaling without LBFGS.

Given raw logits z and 0/1 labels y, find the temperature T > 0 that minimises the log loss of
sigmoid(z / T). Use any method you like EXCEPT scipy/torch LBFGS: a grid search over log T, or
plain gradient descent on log T, are both fine. Return a float.

Then answer (LEARN.md, question 3): can temperature scaling change accuracy? Why or why not?
"""
import numpy as np


def fit_temperature(z: np.ndarray, y: np.ndarray) -> float:
    raise NotImplementedError("implement me")
