import sys
from glob import glob
import pandas as pd

files = glob(r'/storage/users/g-and-n/plates/csvs/*')
files.sort()
print(len(files))  # 406

file_id = int(sys.argv[1])
file_path = files[file_id]

df: pd.DataFrame = pd.read_csv(file_path)
df = df.set_index(
    ['Plate', 'Metadata_ASSAY_WELL_ROLE', 'Metadata_broad_sample', 'Image_Metadata_Well', 'ImageNumber', 'ObjectNumber'])
df.to_csv(file_path)
