import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import librosa 
import pandas as pd
import os
import argparse

# Audio params
SAMPLE_RATE = 22050  # (samples/sec)
DURATION = 10.0  # duration in second (sec)
AUDIO_LEN = int(SAMPLE_RATE * DURATION)  # total number of samples in DURATION

# MFCC params
N_MELS = 128  # freq axis, number of filters
N_FFT = 2048  # frame size
HOP_LEN = 512  # non-overlap region, which means 1/4 portion overlapping
SPEC_WIDTH = AUDIO_LEN // HOP_LEN + 1  # time axis
FMAX = SAMPLE_RATE // 2  # max frequency, based on the rule, it should be half of SAMPLE_RATE
SPEC_SHAPE = [N_MELS, SPEC_WIDTH]  # expected output spectrogram shape


def load_audio(filepath, sr=SAMPLE_RATE):  # load the audio
    audio, sr = librosa.load(filepath, sr=sr)
    return audio, sr

def get_mfcc(audio, sr=SAMPLE_RATE):    # calculate MFCC
    audio_emphasis = librosa.effects.preemphasis(audio, coef=0.97)  # balanced
    mfcc = librosa.feature.mfcc(y=audio_emphasis, sr=sr, n_mfcc=13, fmax=FMAX, n_mels=N_MELS, hop_length=HOP_LEN, n_fft=N_FFT)
    return mfcc

def plot_mfcc(mfcc, sr=SAMPLE_RATE):  # Plot the mel-spectrogram
    fig = librosa.display.specshow(mfcc, x_axis='time', sr=sr, hop_length=HOP_LEN, cmap='viridis')
    return fig

def get_mel_spectrogram(audio, sr=SAMPLE_RATE):  # Get the mel-spectrogram
    spec = librosa.feature.melspectrogram(y=audio, sr=sr, fmax=FMAX, n_mels=N_MELS, hop_length=HOP_LEN, n_fft=N_FFT)
    spec = librosa.power_to_db(spec)  # Turn into log-scale
    return spec

def plot_mel_spectrogram(spec, sr=SAMPLE_RATE):  # Plot the mel-spectrogram
    fig = librosa.display.specshow(spec, sr=sr, hop_length=HOP_LEN, cmap='viridis')
    return fig

def mfcc_csv(audio_source_folder, csv_filename, np_folder):
    # 讀取資料夾內的所有音檔
    audio_files = [f for f in os.listdir(audio_source_folder) if f.endswith(('.wav', '.mp3'))]

    results = []
    for audio_file in audio_files:

        print(f"Current processing file: {audio_file}")

        # 完整的音檔路徑
        file_path = os.path.join(audio_source_folder, audio_file)
        audio, sr = load_audio(file_path, sr=SAMPLE_RATE)

        # divide the segment
        interval = np.ceil(len(audio) / AUDIO_LEN)  # at least 1 segment
        segments = np.array_split(audio, interval)  # Divide audio into `interval` segments
        for i, segment in enumerate(segments):

            # padding audio with original content
            if len(segment) < AUDIO_LEN:
                length_segment = len(segment)
                repeat_count = (AUDIO_LEN + length_segment - 1) // length_segment  # Calculate the `ceiling` of AUDIO_LEN / length_segment
                segment = np.tile(segment, repeat_count)[:AUDIO_LEN]  # Repeat and cut to the required length
            else:
                segment = segment[:AUDIO_LEN]

            # calculate features
            features = get_mfcc(segment, sr=SAMPLE_RATE)
            # mfcc stored in np_file
            if not os.path.exists(np_folder):
                os.makedirs(np_folder)
            np_file = os.path.join(np_folder, f"{audio_file[:-4]}_segment{i}.npy")
            np.save(np_file, features)

            # 將 filename, label, np_file 存入results
            # audio_file: {label}_{filename}
            results.append([f"{audio_file[:-4]}" + f"_segment{i}.png"] + [audio_file[0]] + [np_file])

    # Convert the results to a pandas DataFrame
    columns = ['Audio_name'] + ['Label'] + ['MFCC_file']
    df = pd.DataFrame(results, columns=columns)

    # If file does not exist, write a new CSV file
    df.to_csv(csv_filename, index=False)

def mel_spec_image(audio_source_folder, image_folder):
    audio_files = [f for f in os.listdir(audio_source_folder) if f.endswith(('.wav', '.mp3'))]

    for audio_file in audio_files:
        print(f"Processing mel spectrogram for: {audio_file}")

         # 完整的音檔路徑
        file_path = os.path.join(audio_source_folder, audio_file)
        audio, sr = load_audio(file_path, sr=SAMPLE_RATE)

        # divide the segment
        interval = np.ceil(len(audio) / AUDIO_LEN)  # at least 1 segment
        segments = np.array_split(audio, interval)  # Divide audio into `interval` segments
        for i, segment in enumerate(segments):

            # padding audio with original content
            if len(segment) < AUDIO_LEN:
                length_segment = len(segment)
                repeat_count = (AUDIO_LEN + length_segment - 1) // length_segment  # Calculate the `ceiling` of AUDIO_LEN / length_segment
                segment = np.tile(segment, repeat_count)[:AUDIO_LEN]  # Repeat and cut to the required length
            else:
                segment = segment[:AUDIO_LEN]

            # get mel-spec features
            mel_spectrogram = get_mel_spectrogram(audio=segment, sr=SAMPLE_RATE)
            fig = plot_mel_spectrogram(mel_spectrogram)
            plt.axis('off')

            if not os.path.exists(image_folder):
                os.makedirs(image_folder)

            image_file = os.path.join(image_folder, f"{audio_file[:-4]}_segment{i}.png")
            plt.savefig(image_file, bbox_inches='tight', pad_inches=0)  # remove white background
            plt.close() # free up memory resources

if __name__ == "__main__":
    # Set up argument parser
    parser = argparse.ArgumentParser(description='Extract audio features')
    parser.add_argument('--audio_folder', default='./dataset/audios', help='Folder containing audio files')
    parser.add_argument('--csv_output', default='./dataset/mfcc_features.csv', help='Output CSV file path')
    parser.add_argument('--np_folder', default='./dataset/mfcc_features', help='Folder for numpy feature files')
    parser.add_argument('--image_folder', default='./dataset/mel_spectrograms', help='Folder for mel spectrograms')
    parser.add_argument('--mfcc', action='store_true', help='Extract MFCC features')
    parser.add_argument('--mel', action='store_true', help='Generate mel spectrograms')
    
    args = parser.parse_args()
    
    # Execute based on arguments
    if args.mfcc:
        print(f"Starting to extract MFCC features from {args.audio_folder}...")
        mfcc_csv(args.audio_folder, args.csv_output, args.np_folder)
        print("MFCC extraction completed")
        
    if args.mel:
        print(f"Generating mel spectrograms from {args.audio_folder}...")
        mel_spec_image(args.audio_folder, args.image_folder)
        print("Spectrogram generation completed")
        
    # If no action specified, show help
    if not (args.mfcc or args.mel):
        parser.print_help()