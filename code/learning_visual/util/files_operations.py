import os
import pickle
import json
import tarfile
import csv
import yaml
import gzip
import shutil
import zipfile

import pandas as pd


def is_file_exist(file_path):
    return os.path.isfile(file_path)


def is_folder_exist(folder_path):
    return os.path.isdir(folder_path)


def get_current_working_directory():
    return os.getcwd()


def get_file_name_from_file_path(file_path):
    return os.path.basename(file_path)


def get_folder_path_from_file_path(file_path):
    return os.path.dirname(file_path)


def delete_folder(folder_path):
    shutil.rmtree(folder_path)


def delete_file(file_path):
    os.remove(file_path)


def rename_file(full_file_path, new_file_name):
    new_full_path = os.path.join(os.path.split(full_file_path)[0], new_file_name)
    os.rename(full_file_path, new_full_path)


def make_folder(folder_path):
    os.makedirs(folder_path, exist_ok=True)


def make_folder_for_file_creation_if_not_exists(file_path):
    folder_path = os.path.dirname(file_path)
    if folder_path:
        make_folder(folder_path)


def save_to_json(data, file_path):
    make_folder_for_file_creation_if_not_exists(file_path)
    with open(file_path, 'w') as outfile:
        json.dump(data, outfile)
        outfile.close()


def save_to_pickle(obj, file_path):
    make_folder_for_file_creation_if_not_exists(file_path)
    with open(file_path, 'wb') as handle:
        pickle.dump(obj, handle, protocol=pickle.HIGHEST_PROTOCOL)
        handle.close()


def write_list_of_elements_to_text_file(lines, file_path):
    make_folder_for_file_creation_if_not_exists(file_path)
    text_file = open(file_path, "w")
    for line in lines:
        text_file.write(str(line) + "\n")

    text_file.close()


def write_dict_to_csv_with_pandas(data, file_path, fields_order=None):
    make_folder_for_file_creation_if_not_exists(file_path)
    if type(data[max(data.keys())]) == list:
        expected_length = data[max(data.keys())].__len__()
    else:
        expected_length = 1

    for key in data:
        if data[key].__len__() != expected_length:
            print(
                "Key: " + key + ", Length: " + str(data[key].__len__()) + ", expected length: " + str(expected_length))
            raise ValueError("Keys don't have the same length")

    pd.DataFrame(data).to_csv(file_path, columns=fields_order, index=None)


def load_json(file_path):
    with open(file_path) as f:
        json_file = json.load(f)
        f.close()
    return json_file


def load_multiline_json(file_path):
    entries = []
    with open(file_path) as f:
        for line in f:
            entries.append(json.loads(line))
        f.close()

    return entries


def load_pickle(file_path):
    with open(file_path, 'rb') as handle:
        pkl_file = pickle.load(handle)
        handle.close()
    return pkl_file


def load_yaml(file_path):
    with open(file_path, 'r') as stream:
        return yaml.safe_load(stream)


def load_csv(file_path, as_dataframe=False):
    with open(file_path) as f:
        output = [{k: v for k, v in row.items()}
                  for row in csv.DictReader(f, skipinitialspace=True)]
    if as_dataframe:
        return pd.DataFrame(output)
    else:
        return output


def read_text_file_lines(file_path):
    file = open(file_path, 'r')
    lines = file.readlines()
    file.close()

    return lines


def compress_folder_to_xz(dir_path):
    prev_dir = os.path.abspath(os.curdir)

    par_dir = os.path.dirname(os.path.abspath(dir_path))
    os.chdir(dir_path)
    dir_name = os.path.basename(dir_path)

    with tarfile.TarFile.xzopen(
            os.path.join(par_dir, dir_name + '.tar.xz'), 'w', preset=9
    ) as tar_xz:
        for name in os.listdir(''):
            tar_xz.add(name)

    os.chdir(prev_dir)


def compress_file(file_path):
    make_folder_for_file_creation_if_not_exists(file_path)
    with open(file_path, 'rb') as f_in, gzip.open(file_path + '.gz', 'wb') as f_out:
        shutil.copyfileobj(f_in, f_out)


def compress_files_to_zip_archive(file_paths_to_compress, zip_path):
    zip_obj = zipfile.ZipFile(zip_path, 'w')
    for file_path in file_paths_to_compress:
        zip_obj.write(file_path, compress_type=zipfile.ZIP_DEFLATED)

    zip_obj.close()
