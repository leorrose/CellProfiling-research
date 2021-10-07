import os
from multiprocessing import Pool

import pandas as pd
from tqdm import tqdm

mt_fld = '/storage/users/g-and-n/plates/metadata'
img_fld = '/storage/users/g-and-n/plates/images'

mt_plates = {p.split('.')[0] for p in os.listdir(mt_fld)}
img_plates = set(os.listdir(img_fld))

both_plates = mt_plates & img_plates

if mt_plates - both_plates:
    print(f'There are no images for {" ".join(mt_plates - both_plates)}')

if img_plates - both_plates:
    print(f'There are no metadata for {" ".join(img_plates - both_plates)}')


def check_plate(p):
    mt_path = os.path.join(mt_fld, f'{p}.csv')
    mt_df = pd.read_csv(mt_path)

    for i, row in mt_df.iterrows():
        for c in ['AGP', 'DNA', 'ER', 'Mito', 'RNA']:
            img_name = row.get(c)
            img_path = os.path.join(img_fld, p, img_name)
            if not os.path.exists(img_path):
                return p

    return None


p = Pool(5)

bad_plates = p.map(check_plate, both_plates)
p.close()
p.join()

bad_plates = [p for p in bad_plates if p]

print()
if bad_plates:
    print(f'Bad Plates: {" ".join(bad_plates)}')
else:
    print('There are no bad plates')
