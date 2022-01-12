from abc import ABC, abstractmethod

import numpy as np
import pandas as pd


class ReproducibilityStrategy(ABC):

    @abstractmethod
    def reproduce_measure_df(self, compounds: pd.DataFrame):
        pass

    @abstractmethod
    def reproduce_measure_pair(self, compound1, compound2):
        pass


class MeanDistanceStrategy(ReproducibilityStrategy):
    def reproduce_measure_df(self, compounds: pd.DataFrame):
        pass

    def reproduce_measure_pair(self, compound1, compound2):
        c_name1, c_vals1 = compound1
        c_name2, c_vals2 = compound2
        center1, center2 = c_vals1.mean(), c_vals2.mean()
        return np.abs(center1 - center2)


class UniquenessStrategy(ReproducibilityStrategy):
    def reproduce_measure_df(self, compounds: pd.DataFrame):
        return compounds.index.unique(1).shape[0] / compounds.shape[0] if compounds.shape[0] else np.nan

    def reproduce_measure_pair(self, compound1, compound2):
        pass


class VectorDistanceStrategy(ReproducibilityStrategy):
    def reproduce_measure_df(self, compounds: pd.DataFrame):
        pass

    def reproduce_measure_pair(self, compound1, compound2):
        c_name1, c_vals1 = compound1
        c_name2, c_vals2 = compound2

        return np.linalg.norm(c_vals1 - c_vals2, ord=2)


class SpecificityStrategy(ReproducibilityStrategy):
    def reproduce_measure_df(self, compounds: pd.DataFrame):
        pass

    def reproduce_measure_pair(self, compound1, compound2):
        c_name1, c_vals1 = compound1
        c_name2, c_vals2 = compound2

        center1, center2 = c_vals1.mean(), c_vals2.mean()
        adj_c_vals2 = c_vals2 + (center1 - center2)
        return np.linalg.norm(c_vals1 - adj_c_vals2, ord=1)
