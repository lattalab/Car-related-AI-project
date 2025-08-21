import numpy as np
import librosa
import time
import os
import sys
import noisereduce as nr
import math
import soundfile as sf  # 用於保存音訊

SAMPLE_RATE = 22050  # (samples/sec)
N_FFT = 2048  # frame size
HOP_LEN = 512  # non-overlap region
noise_factor = 0.01
SNR = 20
speed_factor = 1.5
shift_amount = SAMPLE_RATE // 2

# use librosa to load audio file
def load_audio(file_path, sr=SAMPLE_RATE):
    y, sr = librosa.load(file_path, sr=sr)
    return y, sr

# add noise by calculating the signal-to-noise ratio
def get_white_noise(signal, SNR):
    RMS_s = math.sqrt(np.mean(signal ** 2))
    RMS_n = math.sqrt(RMS_s ** 2 / (10 ** (SNR / 10)))
    STD_n = RMS_n
    noise = np.random.normal(0, STD_n, signal.shape[0])
    return signal + noise

# add noise to audio file
def noise_addition(data, noise_factor):
    noise = np.random.randn(len(data))
    augmented_data = data + noise_factor * noise
    return augmented_data.astype(data.dtype)

# pitch modification
def pitch_modified(data, sampling_rate=SAMPLE_RATE, n_steps=8):
    return librosa.effects.pitch_shift(data, sr=sampling_rate, n_steps=n_steps)

# speed change
def speed_change(data, speed_factor=1.0):
    return librosa.effects.time_stretch(data, rate=speed_factor)

# time shift
def time_shift(data, shift_amount):
    return np.roll(data, shift_amount)

# dictionary for data augmentation
wayForDataAug = {
    "noise": get_white_noise,
    "pitch": pitch_modified,
    "speed": speed_change,
    "shift": time_shift
}

def process_audio_file(filepath, dataAug, destination_folder):
    start_time = time.time()  # 計時
    audio, sr = librosa.load(filepath, sr=SAMPLE_RATE)  # 讀取音檔

    # # Apply noise reduction
    # if str(dataAug) != 'noise':
    #     audio = nr.reduce_noise(y=audio, sr=SAMPLE_RATE, stationary=True, prop_decrease=0.9, n_fft=N_FFT, hop_length=HOP_LEN)

    way = wayForDataAug.get(dataAug, None)
    if way == get_white_noise:
        audio = way(audio, SNR)
    elif way == pitch_modified:
        audio = way(audio, SAMPLE_RATE)
    elif way == speed_change:
        audio = way(audio, speed_factor)
    elif way == time_shift:
        audio = way(audio, shift_amount)
    else:
        print(f"Unknown data augmentation method: {dataAug}")
        return

    # Save the processed audio file
    filename = os.path.split(filepath)[-1]
    save_filepath = os.path.join(destination_folder, str(dataAug) + "_" + filename)
    sf.write(save_filepath, audio, SAMPLE_RATE)

    end_time = time.time()
    print(f"Processed {os.path.basename(filepath)} in {end_time - start_time:.2f} seconds")

if __name__ == "__main__":
    if len(sys.argv) != 4:
        print("Usage: python DataAugmentation.py <source_folder> <destination_folder> <dataAugmentation>")
        print("dataAugmentation options: noise, pitch, speed, shift")
        sys.exit(1)

    source_folder = sys.argv[1]
    destination_folder = sys.argv[2]
    dataAug = sys.argv[3]

    os.makedirs(destination_folder, exist_ok=True)  # 存放處理後音檔的資料夾
    files = os.listdir(source_folder)
    for file in files:
        if not file.endswith(".wav"):
            continue
        print("Processing audio file:", file)
        filepath = os.path.join(source_folder, file)  # 從這邊讀取音檔
        process_audio_file(filepath, dataAug, destination_folder)

    print(f"Processed audio files saved to {destination_folder}")
