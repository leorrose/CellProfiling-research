from os import path, scandir, makedirs, remove
from sys import argv
from random import sample
from ftplib import FTP
import re
from progressbar import Bar, ETA,  FileTransferSpeed, Percentage, ProgressBar
import tarfile
from tqdm import tqdm
import pandas as pd
import sqlite3

# Ftp constants
FTP_LINK = r"parrot.genomics.cn/gigadb/pub/10.5524/100001_101000/100351"
TIMEOUT = 10  # In seconds
MAX_ATTEMPTS = 6  # Number of Retries upon timeout


def merger(directory, plate_folders, destination):
    makedirs(destination, exist_ok=True)
    dir_list = [f.name for f in scandir(directory) if f.is_dir()]
    dir_list = [fld for fld in dir_list if fld in plate_folders]

    p_bar = tqdm(dir_list)
    for plate_folder in p_bar:
        p_bar.set_description("Merging {}".format(plate_folder))
        folder_path = path.join(directory, plate_folder)
        plate_number = plate_folder.split('_')[1]
        output = path.join(destination, plate_number+".csv")
        if path.lexists(output):
            print("Warning: {} already merged, skipping...".format(plate_folder))
            continue

        sql_file = path.join(folder_path, plate_number+".sqlite")
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
        df_join.to_csv(output, index=False)


def extractor_file(plate_file, destination):
    plate_basename = path.basename(plate_file)
    plate_name = plate_basename.split(".")[0]
    plate_number = plate_name.split("_")[1]

    sql_file = r"gigascience_upload/{}/extracted_features/{}.sqlite".format(plate_name, plate_number)
    profile_file = r"gigascience_upload/{}/profiles/mean_well_profiles.csv".format(plate_name)

    tar = tarfile.open(plate_file, "r:gz")

    for infile in [sql_file, profile_file]:
        tar_member = tar.getmember(infile)
        tar_member.name = path.basename(infile)
        curr_dest = path.join(destination, plate_name)
        extracted_file = path.join(curr_dest, tar_member.name)
        if path.lexists(extracted_file):
            print("Warning: {}/{} already extracted, skipping...".format(plate_name, tar_member.name))
            continue
        tar.extract(tar_member, curr_dest)

    tar.close()


def extractor(tars_dir, plate_list, destination):
    makedirs(destination, exist_ok=True)
    tars = [f.name for f in scandir(tars_dir) if f.is_file()]
    tars = [tar for tar in tars if tar in plate_list]
    p_bar = tqdm(tars)
    for tar in p_bar:
        p_bar.set_description("Extracting {}".format(tar))
        extractor_file(path.join(tars_dir, tar), destination)


def download_plates(ftp, destination, plate_list):
    makedirs(destination, exist_ok=True)

    for plate in plate_list:
        dest = path.join(destination, plate)
        if path.lexists(dest):
            print("Warning: {} already exist, skipping..".format(plate))
            continue

        ftp = download_file(ftp, plate, dest)

    ftp.quit()


def plate_selector(plate_amount, plate_numbers):
    if plate_numbers:
        reg = r"({})".format("|".join(plate_numbers))
    else:
        reg = r"\d{5}"
    fmt = r"^Plate_{}.tar.gz$".format(reg)
    pattern = re.compile(fmt)
    ftp = connect_ftp()
    plate_list = [plate for plate in ftp.nlst() if pattern.fullmatch(plate)]
    if plate_amount and plate_amount < len(plate_list):
        plate_list = sample(plate_list, plate_amount)
    return ftp, plate_list


def download_file(ftp, plate, dest_file):
    # https://stackoverflow.com/questions/51684008/show-ftp-download-progress-in-python-progressbar
    widgets = ['Downloading: %s ' % plate, Percentage(), ' ',
               Bar(marker='█', left='[', right=']'),
               ' ', ETA(), ' ', FileTransferSpeed()]
    size = ftp.size(plate)
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
            if attempts_left:
                attempts_left -= 1
                print(" Got {} Retry #{}".format(timeout_ex, MAX_ATTEMPTS - attempts_left))
                ftp = connect_ftp()
            else:
                print(" Failed to download {}".format(plate))
                cur_file.close()
                del cur_file
                remove(dest_file)
                prog_bar.finish(dirty=True)
                break
    return ftp


def connect_ftp():
    ftp_split = FTP_LINK.split(r"/")
    ftp_domain = ftp_split[0]
    ftp = FTP(ftp_domain, timeout=10)
    ftp.login()
    if len(ftp_split) > 1:
        ftp_cwd = "/".join(ftp_split[1:])
        ftp.cwd(ftp_cwd)
    return ftp


def main(working_path, plate_amount, plate_numbers):
    ftp, plate_list = plate_selector(plate_amount, plate_numbers)

    download_path = path.join(working_path, "tars")
    download_plates(ftp, download_path, plate_list)

    extract_path = path.join(working_path, "extracted")
    extractor(download_path, plate_list, extract_path)

    merge_path = path.join(working_path, "csvs")
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
        if not argv[i+1].isnumeric():
            print("plate_amount has to be a valid number")
            exit(-1)
        else:
            plate_amount = int(argv[i+1])
            i += 2

    if i < len(argv)-1 and argv[i] == "-l":
        if not valid_numbers(argv[i+1:]):
            print("plate numbers have to be valid numbers")
            exit(-1)
        else:
            plate_numbers = argv[i+1:]

    main(argv[1], plate_amount, plate_numbers)


