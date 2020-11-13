from itertools import cycle
from multiprocessing import Pool, cpu_count

import pandas as pd
from os import scandir, path, makedirs, chdir
from sys import argv
from tqdm import tqdm
import matplotlib.pyplot as plt
from learning.constants import CHANNELS, LABEL_FIELD

# Warning - need openpyxl package
from learning.preprocessing import list_columns, load_plate_csv


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
            df = pd.read_csv(src, index_col=[LABEL_FIELD, 'Metadata_broad_sample', 'Image_Metadata_Well', 'ImageNumber',
                                             'ObjectNumber'])

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


def features_stats(csv_path, dest):
    chdir(csv_path)

    csv_list = [f.name for f in scandir()
                if f.is_file() and f.name.endswith('.csv')]

    print("Process plates' csv")
    p = Pool(cpu_count()-1)

    results = p.starmap(extract_stats_from_plate, zip(csv_list, cycle([dest])))
    p.close()
    p.join()

    print("Process plates' summery")
    plates_summery = [result[0] for result in results]
    plates_summery = pd.concat(plates_summery)
    plates_summery.to_csv(path.join(dest, 'CW_Plates_Summery.csv'))

    stats_per_plate = [result[1] for result in results]
    stats_per_plate = {stat: [dic[stat] for dic in stats_per_plate] for stat in stats_per_plate[0]}
    stats_per_plate = {stat: pd.concat(stats_per_plate[stat]) for stat in stats_per_plate}

    print("Process plates' summery")
    p = Pool(cpu_count()-1)
    p.starmap(extract_stats_all_plates,
                    zip(stats_per_plate.keys(), stats_per_plate.values(), cycle([dest])))
    p.close()
    p.join()


def extract_stats_from_plate(csv, dest):
    print(f'Processing: {csv}')
    plate_number = csv.split(".")[0]

    plate_folder = path.join(dest, plate_number)
    makedirs(plate_folder, exist_ok=True)

    df = load_plate_csv(csv)

    gen_cols, corr_cols, channel_dict = list_columns(df)
    df.drop(corr_cols, axis=1, inplace=True)
    df = df[df.index.isin(['mock'], 1)]

    gb = df.groupby(['Plate', 'Image_Metadata_Well'])

    desc = gb.describe()
    desc.to_csv(path.join(plate_folder, f'CW_{plate_number}_Summery.csv'))

    for stat in desc.columns.unique(1):
        nest_desc = desc.xs(stat, level=1, axis=1).describe()
        nest_desc.to_csv(path.join(plate_folder, f'CW_{plate_number}_{stat}.csv'))

    desc = df.groupby(['Plate']).describe()

    return desc, {stat: desc.xs(stat, level=1, axis=1) for stat in desc.columns.unique(1)}


def extract_stats_all_plates(stat: str, df: pd.DataFrame, dest: str):
    desc = df.describe()
    desc.to_csv(path.join(dest, f'CW_Plate_{stat}.csv'))


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
    features_stats(csvs_folder, output_folder)
