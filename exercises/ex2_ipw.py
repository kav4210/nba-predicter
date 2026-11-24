"""Exercise 2: inverse-probability weighting from scratch.

Given outcomes y, binary treatment t, and propensity scores p = P(t=1 | X), return the
NORMALISED (Hajek) IPW estimate of the average treatment effect:

    sum(t*y/p) / sum(t/p)  -  sum((1-t)*y/(1-p)) / sum((1-t)/(1-p))

Then answer in your own words (LEARN.md, question 5): why does weighting by 1/p remove confounding,
and what goes wrong when some p are near 0 or 1?
"""
import numpy as np


def ipw_ate(y: np.ndarray, t: np.ndarray, p: np.ndarray) -> float:
    raise NotImplementedError("implement me")
