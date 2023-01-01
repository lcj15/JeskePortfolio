"""top level module for classifier abstractions
__author__ = Dr. Soumya Ray, Case Western Reserve University
"""
from abc import ABC, abstractmethod
from typing import Optional

import numpy as np
import util


class Classifier(ABC):
    """Abstract base class defining common classifier functions.
    Classifier implementations should inherit this class, not instantiate it directly.
    """

    @abstractmethod
    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        """fit the classifier given a set of examples X with shape (num_examples, num_features) and labels y with shape (num_examples,).
        Args:
            X (np.ndarray): the example set with shape (num_examples, num_features)
            y (np.ndarray): the labels with shape (num_examples,)
            weights (Optional[np.ndarray]): the example weights, if necessary
        """
        pass

    @abstractmethod
    def predict(self, X: np.ndarray) -> np.ndarray:
        """produce a list of output labels for a set of examples X with shape (num_examples, num_features).
        Args:
            X (np.ndarray): examples for which outputs should be provided
        Returns:
            np.ndarray: the predicted outputs with shape (num_examples,)
        """
        pass

    # todo make documentation more consistent (should look like above)
    def score(self, X: np.ndarray, y: np.ndarray):
        """
        Return the accuracy of the model
        Args:
            X: testing data
            y: testing labels
        Returns:
            Accuracy of the prediction of the model
        """
        return util.accuracy(self.predict(X), y)
