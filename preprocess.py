import numpy as np
import librosa
import soundfile as sf
from tqdm import tqdm

def extract_intervals(fn, allowed={'I0', 'I1', 'I2', 'U0', 'U1', 'U2'}):
    """
    Extract relevant time intervals from a transcription file.
    """
    chi_started = None
    all_intervals = []
    
    for line in open(fn, 'r'):
        line = line.strip()
        
        if line.lower() == '@end':
            break
        
        if not line or line.startswith('@'):
            continue

        if line.startswith('*CHI:\tv@.'):
            try:
                chi_started = [int(i) for i in line[:-1].split('_')[-2:]]
            except (ValueError, IndexError):
                print("parsing error " + line)
                pass
        
        elif line.startswith(('*MOT:\t', '*FAT:\t')):
            chi_started = None

        elif line.startswith('%voc:\t'):
            code = line.split('\t')[-1].strip()

            if chi_started and code in allowed:
                all_intervals.append(chi_started)
                chi_started = None

    return np.array(all_intervals)


def apply_intervals(wave, rate, intervals):
    """
    Marks relevance of each timestep in a wave
    """    
    mark_incr = 1000 / rate
    timesteps = np.zeros(len(wave))

    for interval in intervals:
        first = int(interval[0] / mark_incr)
        last = int(interval[1] / mark_incr)
        timesteps[first:last] = 1

    return timesteps


def mark_spectro(spectro_length, num_frames,
                 hop_length, markers, threshold=0.5):
    """
    function that calculates the overlap from the previously computed markers
    """
    overlaps = np.array([markers[i:(i + num_frames)].mean()
                for i in range(0, len(markers), hop_length)])
        
    overlaps[overlaps >= threshold] = 1
    overlaps[overlaps < threshold] = 0
    
    overlaps = overlaps[:spectro_length]

    return overlaps


def invert_spectrogram(spectrogram, fft_size, hop):
    """
    out of log > de-apply mel filters > convert spectrogram > normalize
    """
    x_reconstruct = audio_utilities.reconstruct_signal_griffin_lim(spectrogram,
                                                                   fft_size,
                                                                   hopsamp=hop,
                                                                   iterations=1000)
    max_sample = np.max(abs(x_reconstruct))
    if max_sample > 1.0:
        x_reconstruct = x_reconstruct / max_sample
    return x_reconstruct


def read_audio(audio_path, target_fs=None):
    #(audio, fs) = librosa.load(audio_path, sr=None)
    (audio, fs) = sf.read(audio_path)
    # if this is not a mono sounds file
    if audio.ndim > 1:
        audio = np.mean(audio, axis=1)
    if target_fs is not None and fs != target_fs:
        audio = librosa.resample(audio, orig_sr=fs, target_sr=target_fs)
        fs = target_fs
    return audio, fs

def griffinlim(spectrogram, n_iter=50, window='hann', n_fft=2048, win_length=2048, hop_length=-1, verbose=False):
    # https://github.com/barronalex/Tacotron/blob/master/audio.py
    if hop_length == -1:
        hop_length = n_fft // 4

    angles = np.exp(2j * np.pi * np.random.rand(*spectrogram.shape))

    t = tqdm(range(n_iter), ncols=100, mininterval=2.0, disable=not verbose)
    for i in t:
        full = np.abs(spectrogram).astype(np.complex) * angles
        inverse = librosa.istft(full, hop_length = hop_length, win_length = win_length, window = window)
        rebuilt = librosa.stft(inverse, n_fft = n_fft, hop_length = hop_length, win_length = win_length, window = window)
        angles = np.exp(1j * np.angle(rebuilt))

        if verbose:
            diff = np.abs(spectrogram) - np.abs(rebuilt)
            t.set_postfix(loss=np.linalg.norm(diff, 'fro'))

    full = np.abs(spectrogram).astype(np.complex) * angles
    inverse = librosa.istft(full, hop_length = hop_length, win_length = win_length, window = window)

    return inverse
