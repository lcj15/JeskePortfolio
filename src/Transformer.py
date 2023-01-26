"""top level module for classifier abstractions
__author__ = Liam Jeske
"""
from abc import ABC, abstractmethod
from typing import Optional
import Unique



import typing
from dataclasses import astuple, dataclass, field
from enum import Enum, IntEnum, unique
from typing import Any, Iterable, List, Optional, Sequence, Tuple, Union

import numpy as np
import util


class Transformer(ABC):

    @abstractmethod
    def fit(self, X):
        pass

    @abstractmethod
    def transform(self):
        pass

    @abstractmethod
    def fit_transform(self):
        pass

    @abstractmethod
    def inverse_transform(self):
        pass



"""

class Feature:
    pass


@unique
class FeatureType(Enum):
    #: true/false, 1 or 0, etc
    BINARY = "BINARY"
    #: finite, enumerated set of values, e.g. `['RED', 'GREEN', 'BLUE']`
    NOMINAL = "NOMINAL"
    #: continuous range of values, e.g. real numbers
    CONTINUOUS = "CONTINUOUS"
    #: unique identifier
    ID = "ID"
"""