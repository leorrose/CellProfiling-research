"""This script allows the user to check the csv's of batch one
dataset to make sure conversion was correct.

"""

import os
import pandas as pd
from tqdm import tqdm

# Define project path
PROJ_PATH = "/sise/assafzar-group/assafzar/s-and-l"

# Define path to batch one data
BATCH_1_DATA_PATH = (
    f"{PROJ_PATH}/CellProfiling-research/leor&sarit/"
    "data/batch_one"
)

# Define metadata path
METADATA_PATH = (f"{BATCH_1_DATA_PATH}/metadata.csv")

# Define path to csv files
BATCH_1_SQLITE_PATH = f"{BATCH_1_DATA_PATH}/csv"


def check_data():
  """Method to map the sqlite data.

  """
  # Read metadata
  meta_df = pd.read_csv(METADATA_PATH)

  #
  for metadata_plate in tqdm(meta_df.Metadata_Plate.unique()):
    # Define csv path
    csv_path = f"{BATCH_1_DATA_PATH}/csv/{metadata_plate}.csv"
    # Check if csv does not exist
    if not os.path.isfile(csv_path):
      print(f"{metadata_plate} csv does not exist!")
    # If csv exist
    else:
      # Get number of rows in csv
      csv_num_rows = pd.read_csv(csv_path, usecols=[0]).shape[0]
      # Check number of rows equals to cell count
      expected_num_rows = meta_df[meta_df.Metadata_Plate == metadata_plate
                                 ].cell_count.sum()
      # Check number of rows are as expected
      if csv_num_rows != expected_num_rows:
        print(
            (
                f"{metadata_plate} csv has {csv_num_rows} rows but expected"
                f"{expected_num_rows}!"
            )
        )


if __name__ == "__main__":
  check_data()
