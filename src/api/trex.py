import os
import requests
import zipfile
import logging
from tqdm import tqdm

import src

logger = logging.getLogger("logger")


def extract_trex():
    # Extract the zip file to a directory
    with zipfile.ZipFile(src.DATA_DIR / "trex" / "trex.zip") as zip_file:
        # Get the total number of files to extract
        total_files = len(zip_file.infolist())

        # Set up the progress bar
        progress_bar = tqdm(total=total_files, unit='file', desc='Extracting')

        # Extract all the files in the zip file to the specified directory
        for file in zip_file.infolist():
            zip_file.extract(file, src.DATA_DIR / "trex")
            progress_bar.update(1)

        # Close the progress bar
        progress_bar.close()

    try:
        os.remove(src.DATA_DIR / "trex" / "trex.zip")
    except FileNotFoundError:
        pass


def download_trex():
    # Download the dataset zip file
    url = 'https://ndownloader.figshare.com/files/8760241'
    response = requests.get(url, stream=True)

    total_size = int(response.headers.get('content-length', 0))
    block_size = 1024  # 1 megabyte
    progress_bar = tqdm(total=total_size, unit='B', unit_scale=True, desc='Downloading')

    with open(src.DATA_DIR / "trex" / "trex.zip", 'wb') as output_file:
        for data in response.iter_content(block_size):
            progress_bar.update(len(data))
            output_file.write(data)

    progress_bar.close()

    if total_size != 0 and progress_bar.n != total_size:
        logger.info("Error: the download was incomplete.")
    else:
        logger.info("Download completed.")
