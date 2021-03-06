"""This module holds various development utilities."""
from .validation import check_tensor
from .ports import train_test_split

__all__ = ['check_tensor',
           'train_test_split']
