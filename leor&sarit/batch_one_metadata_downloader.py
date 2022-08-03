"""This script allows the user to download the metadata file of batch one
dataset.

"""

import os
import requests
import gzip
import numpy as np
import pandas as pd
from typing import List, Union
from io import BytesIO, StringIO


def recode_dose(x: float, doses: List[float],
                return_level: bool = False) -> Union[int, float]:
  """ Method to align the doses collected in the dataset to their nearest dose
      point.

  Args:
      x (float): The dose.
      doses (List[float]): Pre defined doses.
      return_level (bool, optional): Flag to return index of corresponding dose
        or the dose itself. Defaults to False.

  Returns:
      Union[int, float]: index of corresponding dose or the dose itself.

  """
  closest_index = np.argmin([np.abs(dose - x) for dose in doses])
  if np.isnan(x):
    return np.NaN
  if return_level:
    return closest_index + 1
  else:
    return doses[closest_index]


def download_metadata() -> None:
  """ Method to download the metadata file.

  """
  # Get dir path of this file
  dir_path = os.path.dirname(os.path.realpath(__file__))

  # Inform user
  print("Downloading started")

  # Defining the zip file URL
  url = (
      "https://github.com/broadinstitute/lincs-cell-painting/blob/master/"
      "profiles/cell_count/2016_04_01_a549_48hr_batch1_metadata_cell_count_"
      "summary.tsv.gz?raw=true"
  )

  # Define file name
  file_name = "/data/metadata.csv"

  # Downloading the file by sending the request to the URL
  req = requests.get(url)

  # Inform user
  print("Downloading completed")
  print("Unzipping started")

  # Extracting the gz file contents
  compressed_file = BytesIO(req.content)
  decompressed_file = gzip.GzipFile(fileobj=compressed_file)

  # Reading contents to data frame
  meta_df = pd.read_csv(StringIO(decompressed_file.read().decode("utf-8")))

  # Inform user
  print("Unzipping completed")
  print("Sorting metadata")

  # Define does mapping
  primary_dose_mapping = [0.04, 0.12, 0.37, 1.11, 3.33, 10, 20]

  # Map doses
  meta_df.mmoles_per_liter = meta_df.mmoles_per_liter.apply(
      lambda x: recode_dose(x, primary_dose_mapping, return_level=False)
  )
  meta_df.drop("mg_per_ml", axis=1, inplace=True)

  # Inform user
  print("Sorting metadata completed")
  print("Writing metadata")

  # Write csv
  meta_df.to_csv(f"{dir_path}/{file_name}", index=False)

  # Inform user
  print("Writing metadata completed")


if __name__ == "__main__":
  download_metadata()
