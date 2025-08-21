# the library for download yt video
from pytubefix import YouTube
from pytubefix.cli import on_progress
import ssl
ssl._create_default_https_context = ssl._create_stdlib_context
# data analysis
import os
import pandas as pd

def read_csv(file_path):
    """
        para: file_path is the location of csv file
        return the url field to download yt video
    """
    df = pd.read_csv(file_path)
    return df['url'], df['video_type']    # read url field, classification type

def download(data: pd.Series, video_types: pd.Series, store_path: str):
    """
        para: data is a pandas Series containing YouTube video URLs
              video_types is a pandas Series containing the classification type
              store_path is the directory where downloaded audio files will be saved
        object: download audio from YouTube videos
    """
    if not os.path.exists(store_path):
        os.makedirs(store_path)
    for i, url in enumerate(data):
        yt = YouTube(url, on_progress_callback = on_progress)
        ys = yt.streams.get_audio_only()    # fetch audio only
        print(f"Downloading {i + 1}/{len(data)}: {yt.title}")
        ys.download(output_path=store_path, filename=f"{video_types[i]}_{yt.title}.wav")

if __name__ == "__main__":
    csv_file = "./dataset/sound_collection.csv"
    output_dir = "./dataset/audios"

    print("Starting to read csv file...")
    urls, video_types = read_csv(csv_file)
    print("Finishing read csv, then starting download...")
    download(urls, video_types, output_dir)
    print("finish")