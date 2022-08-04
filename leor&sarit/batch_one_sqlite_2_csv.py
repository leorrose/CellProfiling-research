"""This script allows the user to convert sqlite to csv's for batch one
dataset.

"""

import os
import pandas as pd
import sqlite3
from typing import Dict
import multiprocessing as mp
from multiprocessing import Pool
from pathlib import Path

# Define project path
PROJ_PATH = "/sise/assafzar-group/assafzar/s-and-l"

# Define metadata path
METADATA_PATH = (
    f"{PROJ_PATH}/CellProfiling-research/"
    "leor&sarit/data/metadata.csv"
)

# Define path to batch one data
BATCH_1_DATA_PATH = (
    f"{PROJ_PATH}/CellProfiling-research/leor&sarit/"
    "data/batch_one"
)

# Define path to sqlite files
BATCH_1_SQLITE_PATH = f"{BATCH_1_DATA_PATH}/sqlite"

# Define chunk size to read csv
CHUNK_SIZE = 300000


def sqlite_2_csv(
    connection_string: str,
    query: str,
    metadata_plate: str,
    well_2_broad_sample: Dict[str, str],
):
  """Method to transform sqlite file to csv file.

  Args:
      connection_string (str): Connection string to sqlite.
      query (str): Sql query to get single cell profiles.
      metadata_plate (str): Plate id.
      well_2_broad_sample (Dict[str, str]): Mapper from well ro broad sample
        to add metadata to csv.

  """
  print(
      (
          f"Process {mp.current_process().name} started working on sqlite "
          f"{metadata_plate}"
      ), flush=True
  )
  # Open connection
  with sqlite3.connect(connection_string, uri=True) as conn:
    # Define csv path
    csv_path = (
        f"{BATCH_1_DATA_PATH}/csv/{metadata_plate}/"
        f"{metadata_plate}.csv"
    )

    # Create dirs if dont exist
    Path(os.path.dirname(csv_path)).mkdir(parents=True, exist_ok=True)

    # Delete csv if exist from previous attempts
    if os.path.isfile(csv_path):
      os.remove(csv_path)

    # Loop over chunks of the db
    for db_df in pd.read_sql_query(query, conn, chunksize=CHUNK_SIZE):

      # Add metadata to each single cell profile
      db_df["Metadata_broad_sample"] = db_df.Image_Metadata_Well.map(
          well_2_broad_sample
      )
      db_df["Metadata_ASSAY_WELL_ROLE"] = db_df["Metadata_broad_sample"].isna(
      ).map({
          True: "treated",
          False: "mock"
      })

      # Rename TableNumber to Plate
      db_df.rename(columns={"TableNumber": "Plate"}, inplace=True)
      db_df["Plate"] = metadata_plate

      # If file does not exist write with the header
      if not os.path.isfile(csv_path):
        db_df.to_csv(csv_path, index=False)

      # Else it exists so append without writing the header
      else:
        db_df.to_csv(csv_path, index=False, mode="a", header=False)

    print(
        (
            f"Process {mp.current_process().name} ended working on sqlite "
            f"{metadata_plate}"
        ), flush=True
    )


def map_data():
  """Method to map the sqlite data.

  """
  # Read metadata
  meta_df = pd.read_csv(METADATA_PATH)

  # Define empty list for plate combinations
  combinations = []

  # Loop over each plate
  for metadata_plate in meta_df.Metadata_Plate.unique():
    # Check if sql folder exist
    if os.path.isdir(f"{BATCH_1_SQLITE_PATH}/{metadata_plate}"):
      # Define connection string to plate db
      connection_string = (
          f"file:{BATCH_1_SQLITE_PATH}/{metadata_plate}"
          f"/{metadata_plate}.sqlite?mode=ro"
      )

      # Define query to get single cell profiles
      query = (
          "SELECT Cells.*, Image.Image_Metadata_Well FROM Cells "
          "INNER JOIN Image ON Cells.TableNumber = Image.TableNumber"
      )

      # Filter metadata for current plate
      plate_meta_df = meta_df.loc[meta_df["Metadata_Plate"] == metadata_plate,
                                  ["Metadata_Well", "broad_sample"]]

      # Create mapper from well to broad sample for this plate
      well_2_broad_sample = dict(
          zip(plate_meta_df.Metadata_Well, plate_meta_df.broad_sample)
      )

      # Append combination
      combinations.append(
          (connection_string, query, metadata_plate, well_2_broad_sample)
      )
    else:
      print(f"{metadata_plate}: No sql folder!")

  # Run multiprocessing to map sqlite to csv
  with Pool(processes=4) as p:
    p.starmap_async(sqlite_2_csv, combinations)
    p.close()
    p.join()


if __name__ == "__main__":
  map_data()
