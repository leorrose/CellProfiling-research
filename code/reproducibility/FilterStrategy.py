from abc import ABC, abstractmethod

import pandas as pd


class FilterStrategy(ABC):

    @abstractmethod
    def filter_dataframe(self, compounds: pd.DataFrame, field: str):
        pass


class AbsoluteFilterStrategy(FilterStrategy):
    def __init__(self, filter_threshold):
        self._filter_threshold = filter_threshold

    def filter_dataframe(self, compounds: pd.DataFrame, field: str):
        df_zero = compounds.query(f'{field} >= 0').index.get_level_values(1).value_counts()
        df_zero.name = 'zero'
        df_filter = compounds.query(f'{field} >= {self._filter_threshold}').index.get_level_values(1).value_counts()
        df_filter.name = 'filter'
        counts = pd.concat([df_zero, df_filter], axis=1).fillna(0).astype(int)
        comp_names = counts.query('zero > 1 and filter > 0').index
        return compounds[compounds.index.isin(comp_names, 1)]


class TopKFilterStrategy(FilterStrategy):
    def __init__(self, top_k):
        self._top_k = top_k

    def filter_dataframe(self, compounds: pd.DataFrame, field: str):
        # return top_k compounds
        return compounds.nlargest(self._top_k, field)
