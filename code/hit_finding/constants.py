import os

os.chdir(f'/storage/users/g-and-n/plates')

model = 'DNN'
err_fld = f'errors/{model}'
raw_fld = 'csvs'
raw1to1_fld = f'errors/{model}1to1'
zsc_fld = f'z_scores/{model}'
pure_fld = f'{zsc_fld}/pure'

plots_path = f'/home/naorko/CellProfiling/plots/fraction-score/{model}'

files = [(int(f.name.split('.')[0]), f.name) for f in os.scandir(err_fld) if f.name.endswith('.csv')]
