import pandas as pd
from os import scandir, path, makedirs
from sys import argv
from tqdm import tqdm
import matplotlib.pyplot as plt
from learning.constants import CHANNELS, LABEL_FIELD


# Warning - need openpyxl package

def extract_statistics(csv_folder, dest):
    csv_list = [f.name for f in scandir(csv_folder)
                if f.is_file() and f.name.endswith('csv')]

    xls_file = path.join(dest, 'Statistics.xlsx')
    with pd.ExcelWriter(xls_file) as xl_writer:
        prog_bar = tqdm(csv_list, desc='Start running')
        for csv in prog_bar:
            prog_bar.set_description('Load {}'.format(csv), refresh=True)
            name = csv.split('.')[0]
            src = path.join(csv_folder, csv)
            df = pd.read_csv(src, index_col=[LABEL_FIELD, 'Metadata_broad_sample', 'Image_Metadata_Well', 'ImageNumber', 'ObjectNumber'])

            prog_bar1 = tqdm(df.columns, desc=f'Plotting features')

            curr_fld = path.join(dest, name)
            makedirs(curr_fld, exist_ok=True)
            for col in prog_bar1:
                # prog_bar1.set_description(f'Plotting {col} in {csv}', refresh=True)
                df[col].plot.hist()
                plt.title(col)
                plt.savefig(path.join(curr_fld, f'{col}.png'))
                plt.close()

            desc = df.describe()
            desc.to_excel(xl_writer, name, freeze_panes=(1, 1))
            del desc

            general_cols = [f for f in df.columns if all(c not in f for c in CHANNELS)]
            corr_cols = [f for f in df.columns if 'Correlation' in f]

            df[general_cols].boxplot(rot=60)
            plt.title(f'Plate_{csv}_General_Features')
            plt.savefig(path.join(dest, f'Plate_{csv}_General_Features.png'))
            plt.close()

            df[corr_cols].boxplot(rot=60)
            plt.title(f'Plate_{csv}_Correlation_Features')
            plt.savefig(path.join(dest, f'Plate_{csv}_Correlation_Features.png'))
            plt.close()

            # Split columns by channel
            for channel in CHANNELS:
                cols = [col for col in df.columns if channel in col and col not in corr_cols]
                df[cols].boxplot(rot=60)
                plt.title(f'Plate_{csv}_Channel_{channel}')
                plt.savefig(path.join(dest, f'Plate_{csv}_Channel_{channel}.png'))
                plt.close()

            del df


if __name__ == '__main__':
    if len(argv) == 2:
        print("Please follow the following usages:")
        print("Usage: extractStatistics.py [csv_folder] [output_folder]")
        exit(-1)

    csvs_folder = argv[1]
    output_folder = argv[2]

    if not path.lexists(csvs_folder):
        print("CSV folder must exist")
        exit(-1)

    makedirs(output_folder, exist_ok=True)
    extract_statistics(csvs_folder, output_folder)
