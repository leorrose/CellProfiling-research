import pandas as pd
from os import scandir, path, makedirs
from sys import argv
from tqdm import tqdm


# Warning - need openpyxl package

def extract_statistics(csv_folder, dest_xls):
    csv_list = [f.name for f in scandir(csv_folder)
                if f.is_file() and f.name.endswith('csv')]

    prog_bar = tqdm(csv_list, desc='Start running')
    with pd.ExcelWriter(dest_xls) as xl_writer:
        for csv in prog_bar:
            prog_bar.set_description('Load {}'.format(csv), refresh=True)
            name = csv.split('.')[0]
            src = path.join(csv_folder, csv)
            df = pd.read_csv(src)
            desc = df.describe().drop(['ImageNumber', 'ObjectNumber'], 'columns')
            del df
            desc.to_excel(xl_writer, name, freeze_panes=(1, 1))
            del desc


if __name__ == '__main__':
    if len(argv) < 2 or len(argv) > 3:
        print("Please follow the following usages:")
        print("Usage: extractStatistics.py [csv_folder]")
        print("Usage: extractStatistics.py [csv_folder] [output_excel_folder]")
        exit(-1)

    csvs_folder = argv[1]
    output_excel_folder = csvs_folder
    if len(argv) == 3:
        output_excel_folder = argv[2]

    if not path.lexists(csvs_folder):
        print("CSV folder must exist")
        exit(-1)

    makedirs(output_excel_folder, exist_ok=True)

    xls_file = path.join(output_excel_folder, 'Statistics.xlsx')
    extract_statistics(csvs_folder, xls_file)
