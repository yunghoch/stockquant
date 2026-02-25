"""Alpha101 - WorldQuant 101 Formulaic Alphas Implementation.

Based on: https://arxiv.org/abs/1601.00991
"101 Formulaic Alphas" by Zura Kakushadze

This module implements all 101 alpha factors for stock prediction.
"""

from .operators import *
from .calculator import Alpha101Calculator
from .relative_strength import RelativeStrengthAlphas

__all__ = ['Alpha101Calculator', 'RelativeStrengthAlphas']
