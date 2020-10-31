# %% Imports:

# Utils
from sys import argv
from os import path, scandir, makedirs, remove
from tqdm import tqdm

# Selector
import re
from random import sample

# Downloader
from ftplib import FTP
from progressbar import Bar, ETA, FileTransferSpeed, Percentage, ProgressBar

# Extractor
import tarfile

# Merger
import pandas as pd
import sqlite3

# Multiprocess
from multiprocessing import Pool
from itertools import cycle


# %% FTP Modifiable constants:

FTP_LINK = r"parrot.genomics.cn/gigadb/pub/10.5524/100001_101000/100351"
TIMEOUT = 10  # In seconds
MAX_ATTEMPTS = 6  # Number of Retries upon timeout
MP_COUNT = 4  # How many parallel processes


# %% Selector:
def plate_selector(plate_amount, plate_numbers, csv_folder):
    """
    Connects to the ftp server and return the list of plates' files
    according to the input parameters

    :param plate_amount: Amount of plates to be selected of None
    :param plate_numbers: List of plates' numbers or None
    :param csv_folder: folder of exists csv, for knowing what not to download
    :return: valid ftp connection and the selected list of plates' files
    """
    if plate_numbers:
        reg = r"({})".format("|".join(plate_numbers))
    else:
        reg = r"\d{5}"
    fmt = r"^Plate_{}.tar.gz$".format(reg)
    pattern = re.compile(fmt)

    ftp = connect_ftp()
    if not ftp:
        exit(-1)

    plate_list = [plate for plate in ftp.nlst() if pattern.fullmatch(plate)]

    if path.exists(csv_folder):
        csv_files = [f.name for f in scandir(csv_folder) if f.is_file and f.name.endswith(".csv")]
        csv_numbers = [f.split('.')[0] for f in csv_files]
        csv_plate_files = [f'Plate_{f}.tar.gz' for f in csv_numbers]
        plate_list = [plate for plate in plate_list if plate not in csv_plate_files]
        dont_download = [plate for plate in plate_list if plate in csv_plate_files]
        if len(dont_download) > 0:
            print(f'Notice: {dont_download} already have refined csv files')

    if plate_amount and plate_amount < len(plate_list):
        plate_list = sample(plate_list, plate_amount)

    return ftp, plate_list


# %% Downloader:

# Connect to the FTP server and returns the connection
def connect_ftp():
    ftp_split = FTP_LINK.split(r"/")
    ftp_domain = ftp_split[0]

    curr_attempts = MAX_ATTEMPTS
    ftp = None
    while not ftp:
        try:
            ftp = FTP(ftp_domain, timeout=TIMEOUT)
            ftp.login()
            if len(ftp_split) > 1:
                ftp_cwd = "/".join(ftp_split[1:])
                ftp.cwd(ftp_cwd)
        except Exception as timeout_ex:
            curr_attempts -= 1
            if ftp:
                ftp.close()

            del ftp
            ftp = None
            if not curr_attempts:
                print(" Could not establish a connection")
                break

            print(" Got {} Retry #{} During connection".format(timeout_ex, MAX_ATTEMPTS - curr_attempts))

    return ftp


# Download a single plate
def download_file(ftp, plate, dest_file):
    try:
        size = ftp.size(plate)
    except Exception as ex:
        print(f"Got {ex} ; Could not retrieve the size of plate {plate}")
        del ftp
        return None

    # https://stackoverflow.com/questions/51684008/show-ftp-download-progress-in-python-progressbar
    widgets = ['Downloading: %s ' % plate, Percentage(), ' ',
               Bar(marker='█', left='[', right=']'),
               ' ', ETA(), ' ', FileTransferSpeed()]
    prog_bar = ProgressBar(widgets=widgets, maxval=size)
    prog_bar.start()
    cur_file = open(dest_file, 'wb')

    def file_write(data):
        cur_file.write(data)
        prog_bar.update(prog_bar.value + len(data))

    # https://stackoverflow.com/questions/8323607/download-big-files-via-ftp-with-python
    attempts_left = MAX_ATTEMPTS
    while size != cur_file.tell():
        try:
            if cur_file.tell():
                ftp.retrbinary("RETR " + plate, file_write, rest=cur_file.tell())
            else:
                ftp.retrbinary("RETR " + plate, file_write)

            cur_file.close()
            prog_bar.finish()
            break
        except Exception as timeout_ex:
            attempts_left -= 1
            if attempts_left:
                print(" Got {} Retry #{}".format(timeout_ex, MAX_ATTEMPTS - attempts_left))
                ftp.close()
                del ftp
                ftp = connect_ftp()
                if ftp:
                    continue

            print(" Failed to download {}".format(plate))
            cur_file.close()
            del cur_file
            remove(dest_file)
            prog_bar.finish(dirty=True)
            break

    return ftp


# Iterate over the plate plist and download them
def download_plates(ftp, destination, plate_list):
    makedirs(destination, exist_ok=True)

    for plate in plate_list:
        dest = path.join(destination, plate)
        if path.lexists(dest):
            print("Warning: {} already exist, skipping..".format(plate))
            continue

        if not ftp:
            print(f" Could not download more plates: {plate_list[plate_list.index(plate):]}")
            break

        ftp = download_file(ftp, plate, dest)

    if ftp:
        ftp.quit()


# %% Extractor:

# Extract a single file
def extractor_file(plate_file, destination):
    print(f'Extracting {path.basename(plate_file)}')
    plate_basename = path.basename(plate_file)
    plate_name = plate_basename.split(".")[0]
    plate_number = plate_name.split("_")[1]

    curr_dest = path.join(destination, plate_name)

    sql_file = r"gigascience_upload/{}/extracted_features/{}.sqlite".format(plate_name, plate_number)
    profile_file = r"gigascience_upload/{}/profiles/mean_well_profiles.csv".format(plate_name)

    tar = tarfile.open(plate_file, "r:gz")

    try:
        for infile in [sql_file, profile_file]:
            tar_member = tar.getmember(infile)
            tar_member.name = path.basename(infile)
            extracted_file = path.join(curr_dest, tar_member.name)
            if path.lexists(extracted_file):
                print(f'Warning: {plate_name}/{tar_member.name} already extracted, skipping...')
                continue
            tar.extract(tar_member, curr_dest)

        tar.close()
        del tar

    except:
        print(f'Warning: {plate_name}.tar.gz corrupted, deleting file...')
        del tar
        remove(plate_file)
        for f in scandir(curr_dest):
            remove(path.join(curr_dest, f.name))
        remove(curr_dest)


# Iterate over the gz files and extract them
def extractor(tars_dir, plate_list, destination):
    makedirs(destination, exist_ok=True)
    tars = [f.name for f in scandir(tars_dir) if f.is_file()]
    tars = [tar for tar in tars if tar in plate_list]
    tars = [path.join(tars_dir, tar) for tar in tars]

    p = Pool(MP_COUNT)

    p_bar = tqdm(desc="Extracting... ", total=len(tars))

    def success(result):
        nonlocal p_bar
        p_bar.update()

    p.starmap_async(extractor_file, zip(tars, cycle([destination])), callback=success)

    p.close()

    p.join()


# %% Merger:
def merger(directory, plate_folders, destination):
    makedirs(destination, exist_ok=True)
    dir_list = [f.name for f in scandir(directory) if f.is_dir()]
    dir_list = [fld for fld in dir_list if fld in plate_folders]

    p = Pool(MP_COUNT)
    p_bar = tqdm(desc="Merging... ", total=len(dir_list))

    def success(result):
        nonlocal p_bar
        p_bar.update()

    p.starmap_async(merge_plate, zip(cycle([destination]), cycle([directory]), dir_list), callback=success)

    p.close()
    p.join()


def merge_plate(destination, directory, plate_folder):
    print("Merging {}".format(plate_folder))
    folder_path = path.join(directory, plate_folder)
    plate_number = plate_folder.split('_')[1]
    output = path.join(destination, plate_number + ".csv")
    if path.lexists(output):
        print("Warning: {} already merged, skipping...".format(plate_folder))
        # continue
    sql_file = path.join(folder_path, plate_number + ".sqlite")
    well_file = path.join(folder_path, "mean_well_profiles.csv")
    df_well = pd.read_csv(well_file,
                          index_col="Metadata_Well",
                          usecols=["Metadata_Well", "Metadata_ASSAY_WELL_ROLE", "Metadata_broad_sample"])
    con = sqlite3.connect(sql_file)
    query = "SELECT Cells.*, Image.Image_Metadata_Well FROM Cells " \
            "INNER JOIN Image ON Cells.ImageNumber = Image.ImageNumber"
    df_cells = pd.read_sql_query(query, con)
    con.close()
    df_join = df_cells.join(df_well, "Image_Metadata_Well", "inner")

    df_join.rename(columns={"TableNumber": "Plate"}, inplace=True)
    df_join["Plate"] = plate_folder.split('_')[1]

    df_join.to_csv(output, index=False)
    del df_well, df_cells, df_join


# %% Main function:
def main(working_path, plate_amount, plate_numbers):
    download_path = path.join(working_path, "tars")
    extract_path = path.join(working_path, "extracted")
    merge_path = path.join(working_path, "csvs")

    ftp, plate_list = plate_selector(plate_amount, plate_numbers, merge_path)

    download_plates(ftp, download_path, plate_list)

    extractor(download_path, plate_list, extract_path)

    plate_folders = [plate.split('.')[0] for plate in plate_list]
    merger(extract_path, plate_folders, merge_path)


def valid_numbers(str_list):
    for st in str_list:
        if not st.isnumeric():
            return False

    return True


if __name__ == '__main__':
    if len(argv) < 4:
        print("Please follow the following usages:")
        print("Usage: plateDownloader.py [working_path] -n [plate_amount]")
        print("Usage: plateDownloader.py [working_path] -l [plate_number1] [plate_number2] ...")
        print("Usage: plateDownloader.py [working_path] -n [plate_amount] -l [plate_number1] [plate_number2] ...")
        exit(-1)

    makedirs(argv[1], exist_ok=True)

    plate_amount = None
    plate_numbers = None

    i = 2
    if argv[i] == "-n":
        if not argv[i + 1].isnumeric():
            print("plate_amount has to be a valid number")
            exit(-1)
        else:
            plate_amount = int(argv[i + 1])
            i += 2

    if i < len(argv) - 1 and argv[i] == "-l":
        if not valid_numbers(argv[i + 1:]):
            print("plate numbers have to be valid numbers")
            exit(-1)
        else:
            plate_numbers = argv[i + 1:]

    main(argv[1], plate_amount, plate_numbers)
