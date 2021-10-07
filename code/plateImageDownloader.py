import os, sys
from multiprocessing import Pool
from urllib import request

CHANNELS = ['Hoechst', 'ERSyto', 'ERSytoBleed', 'Ph_golgi', 'Mito']
dest_path = '/storage/users/g-and-n/plates/images'


def download_plate(plate):
    plate_path = f'{dest_path}/{plate}'
    os.makedirs(plate_path, exist_ok=True)

    for channel in CHANNELS:
        url = f'http://cildata.crbs.ucsd.edu/broad_data/plate_{plate}/{plate}-{channel}.zip'
        dest = f'{plate_path}/{plate}-{channel}.zip'
        request.urlretrieve(url, dest)

        os.system(f'unzip -qjo {dest_path}/{plate}/{plate}-{channel}.zip {plate}-{channel}/* -d {dest_path}/{plate}/')
        os.system(f'rm {dest_path}/{plate}/{plate}-{channel}.zip')

    return None


if __name__ == '__main__':
    if sys.argv[1] == '-i':
        i = int(sys.argv[2])
        with open(r'/storage/users/g-and-n/visual_plates_to_download.txt', 'r') as f:
            plates = [p.strip() for p in f.readlines()]
            plate_numbers = plates[i].split(' ')
    else:
        plate_numbers = sys.argv[1:]

    # p = Pool(8)
    # p.map(download_plate, plate_numbers)
    # p.close()
    # p.join()

    from tqdm import tqdm
    for p in tqdm(plate_numbers):
        print('Downloading', p)
        download_plate(p)
