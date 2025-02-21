
import requests
import zipfile
import os


def download_hapt_dataset(url= 'https://archive.ics.uci.edu/static/public/341/smartphone+based+recognition+of+human+activities+and+postural+transitions.zip', file_name= 'UCI-HAPT.zip'):
    response = requests.get(url)
    with open(file_name, 'wb') as file:
        file.write(response.content)

def unzip_hapt_dataset(file_name= 'UCI-HAPT.zip'):    
    with zipfile.ZipFile(file_name, 'r') as zip_ref:
        zip_ref.extractall('../data/original/UCI')

def __main__():
    download_hapt_dataset()
    unzip_hapt_dataset()

if __name__ == '__main__':
    __main__()
