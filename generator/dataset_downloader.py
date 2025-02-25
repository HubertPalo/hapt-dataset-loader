
import requests
import zipfile
import os


def download_dataset(url, file_name):
    response = requests.get(url)
    with open(file_name, 'wb') as file:
        file.write(response.content)

def download_hapt_dataset(file_name = 'UCI-HAPT.zip'):
    url = 'https://archive.ics.uci.edu/static/public/341/smartphone+based+recognition+of+human+activities+and+postural+transitions.zip'
    download_dataset(url, file_name)

def download_recodgait_v1_dataset(file_name = 'recodgait_v1.zip'):
    url = 'https://figshare.com/ndownloader/files/14502284'
    download_dataset(url, file_name)

def download_recodgait_v2_dataset(file_name = 'recodgait_v2.zip'):
    url = 'https://figshare.com/ndownloader/files/31116205'
    download_dataset(url, file_name)

def unzip_file(zip_file, output_path):
    with zipfile.ZipFile(zip_file, 'r') as zip_ref:
        zip_ref.extractall(output_path)

def unzip_hapt_dataset(file_name= 'UCI-HAPT.zip'):
    unzip_file(file_name, '../data/original/UCI')

def unzip_recodgait_v1_dataset(file_name= 'recodgait_v1.zip'):
    unzip_file(file_name, '../data/original/RecodGait_v1')

def unzip_recodgait_v2_dataset(file_name= 'recodgait_v2.zip'):
    unzip_file(file_name, '../data/original/RecodGait_v2')

def __main__():
    hapt_zip_path = 'UCI-HAPT.zip'
    recodgait_v1_path = 'recodgait_v1.zip'
    recodgait_v2_path = 'recodgait_v2.zip'
    if os.path.exists(hapt_zip_path):
        print('UCI dataset already downloaded')
    else:
        download_hapt_dataset()
    if os.path.exists(recodgait_v1_path):
        print('RecodGait v1 dataset already downloaded')
    else:
        download_recodgait_v1_dataset()
    if os.path.exists(recodgait_v2_path):
        print('RecodGait v2 dataset already downloaded')
    else:
        download_recodgait_v2_dataset()
    
    unzip_hapt_dataset()
    unzip_recodgait_v1_dataset()
    unzip_recodgait_v2_dataset()

if __name__ == '__main__':
    __main__()
