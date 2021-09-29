import os
import pandas as pd
from tqdm import tqdm

mt_fld = '/storage/users/g-and-n/plates/metadata'
img_fld = '/storage/users/g-and-n/plates/images'

mt_plates = {p.split('.')[0] for p in os.listdir(mt_fld)}
img_plates = set(os.listdir(img_fld))

both_plates = mt_plates & img_plates

if mt_plates - both_plates:
    print(f'There are no images for {",".join(mt_plates - both_plates)}')

if img_plates - both_plates:
    print(f'There are no metadata for {",".join(img_plates - both_plates)}')

bad_plates = []
for p in tqdm(both_plates):
    mt_path = os.path.join(mt_fld, f'{p}.csv')
    mt_df = pd.read_csv(mt_path)

    bad_plate = False
    for i, row in mt_df.iterrows():
        for c in ['AGP', 'DNA', 'ER', 'Mito', 'RNA']:
            img_name = row.get(c)
            img_path = os.path.join(img_fld, p, img_name)
            if not os.path.exists(img_path):
                # print(f'\tMissing {img_name} for channel {c}')
                bad_plate = True
                bad_plates.append(p)
                break

        if bad_plate:
            break
print()
if bad_plates:
    print(f'Bad Plates: {bad_plates}')
else:
    print('There are no bad plates')
