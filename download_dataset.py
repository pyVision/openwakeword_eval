
import os
import numpy as np
import torch
import sys
from pathlib import Path
import uuid
import yaml
import datasets
import scipy
from tqdm import tqdm
import os
import argparse
import requests
import zipfile

def downloadMSSNSD(output_dir,skip_download=True):

    
    url="https://github.com/microsoft/MS-SNSD/archive/refs/heads/master.zip"

    os.makedirs(output_dir, exist_ok=True)

    # Path where the file will be saved
    output_path = os.path.join(output_dir, "master.zip")

    if os.path.exists(output_path) and skip_download:
        print(f"File already exists at {output_path}. Skipping download.")

        with zipfile.ZipFile(output_path, 'r') as zip_ref:
            zip_ref.extractall(output_dir)

        # Delete the zip file
        os.remove(output_path)

        return 0

    response = requests.get(url, stream=True)
    total_size = int(response.headers.get('content-length', 0))
    block_size = 1024  # 1 Kibibyte

    with open(output_path, 'wb') as file, tqdm(
        desc=output_path,
        total=total_size,
        unit='iB',
        unit_scale=True,
        unit_divisor=1024,
    ) as bar:
        for data in response.iter_content(block_size):
            file.write(data)
            bar.update(len(data))

    # Unzip the file
    with zipfile.ZipFile(output_path, 'r') as zip_ref:
        zip_ref.extractall(output_dir)

    # Delete the zip file
    os.remove(output_path)

    print(f"File downloaded to {output_path}")
    return 0

def downloadRIR(output_dir):
    #output_dir = "./mit_rirs"
    if not os.path.exists(output_dir):
        os.mkdir(output_dir)
    rir_dataset = datasets.load_dataset("davidscripka/MIT_environmental_impulse_responses", split="train", streaming=True)

    # Save clips to 16-bit PCM wav files
    for row in tqdm(rir_dataset):
        name = row['audio']['path'].split('/')[-1]
        scipy.io.wavfile.write(os.path.join(output_dir, name), 16000, (row['audio']['array']*32767).astype(np.int16))    


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='download datasets')
    parser.add_argument('--output_dir', type=str, required=True, help='Path to the output files')
    parser.add_argument('--dataset', type=str, required=True, help='datasetname')


    args = parser.parse_args()



    if args.dataset=="MIT_RIR":
        downloadRIR(args.output_dir)
    if args.dataset=="MS-SNSD":
        downloadMSSNSD(args.output_dir)
