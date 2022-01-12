import itertools

import numpy as np
import pandas as pd

from reproducibility.FilterStrategy import FilterStrategy
from reproducibility.ReproducibilityStrategy import ReproducibilityStrategy


class ReproducibilityContext:
    def __init__(self, filter_strategy: FilterStrategy, reproducibility_strategy: ReproducibilityStrategy) -> None:
        self._filter_strategy = filter_strategy
        self._reproducibility_strategy = reproducibility_strategy

    @property
    def filter_strategy(self) -> FilterStrategy:
        return self._filter_strategy

    @filter_strategy.setter
    def filter_strategy(self, filter_strategy: FilterStrategy) -> None:
        self._filter_strategy = filter_strategy

    @property
    def reproducibility_strategy(self) -> ReproducibilityStrategy:
        return self._reproducibility_strategy

    @reproducibility_strategy.setter
    def reproducibility_strategy(self, reproducibility_strategy: ReproducibilityStrategy) -> None:
        self._reproducibility_strategy = reproducibility_strategy

    def extract_reproduce_measure(self, compounds: pd.DataFrame) -> None:
        filtered_compounds = self._filter_strategy.filter_dataframe(compounds, 'FIELD')
        measures = self._reproducibility_strategy.reproduce_measure(filtered_compounds)
        return measures

    def compare_triplets(self, compounds, field):
        df_to_pair = self._filter_strategy.filter_dataframe(compounds, field)
        plates_match = df_to_pair.reset_index(0).groupby('Metadata_broad_sample').aggregate(
            {'Plate': lambda int_arr: ','.join([str(x) for x in int_arr])})['Plate'].unique()

        plates = np.unique([int(p) for plates in plates_match for p in plates.split(',')])
        dfs = {k: pd.DataFrame(np.zeros(shape=(len(plates), len(plates))), index=plates, columns=plates)
               for k in ['Metric', 'Triplet Count', 'Pair Count']}

        for plates in plates_match:
            plates = [int(p) for p in plates.split(',')]
            for plate1, plate2 in itertools.combinations(plates, 2):
                par_cnt, cmp_cnt, metric = self.__compare_plates(df_to_pair, plate1, plate2)
                dfs['Triplet Count'].loc[plate1, plate2] += cmp_cnt
                dfs['Pair Count'].loc[plate1, plate2] += par_cnt
                dfs['Metric'].loc[plate1, plate2] += metric
                par_cnt, cmp_cnt, metric = self.__compare_plates(df_to_pair, plate2, plate1)
                dfs['Triplet Count'].loc[plate2, plate1] += cmp_cnt
                dfs['Pair Count'].loc[plate2, plate1] += par_cnt
                dfs['Metric'].loc[plate2, plate1] += metric

        return dfs

    def __compare_plates(self, df_to_pair, plate1, plate2):
        num_of_compares = 0
        num_of_pairs = 0
        acc = 0
        plate1_df, plate2_df = [df_to_pair.loc[[p]] for p in [plate1, plate2]]

        trts1, trts2 = [p_df.index.get_level_values(1) for p_df in [plate1_df, plate2_df]]
        for trt in trts1:
            src_trt = plate1_df.loc[(slice(None), trt), :]
            if trt in trts2:
                num_of_pairs += 1
                same_trt = plate2_df.loc[(slice(None), trt), :]
                same_metric = self._reproducibility_strategy.reproduce_measure_pair(
                    (src_trt.index, src_trt.iloc[0]),
                    (same_trt.index, same_trt.iloc[0]))
                diff_trts = plate2_df.loc[(slice(None), [t for t in trts2 if t != trt]), :]
                num_of_compares += diff_trts.shape[0]

                for diff_trt in diff_trts.iterrows():
                    diff_metric = self._reproducibility_strategy.reproduce_measure_pair(
                        (src_trt.index, src_trt.iloc[0]), diff_trt)

                    if same_metric <= diff_metric:
                        acc += 1

        return num_of_pairs, num_of_compares, acc

    def dataframe_measure(self, compounds: pd.DataFrame, field: str):
        df_filtered = self._filter_strategy.filter_dataframe(compounds, field)
        return self._reproducibility_strategy.reproduce_measure_df(df_filtered)
