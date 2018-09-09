import numpy as np
import librosa
import audio_utilities



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


def load_file(filename, duration=None):
    y, sr = librosa.load(path=filename, sr=None, mono=False)
    mono_signal = librosa.to_mono(y)

    if duration is not None and isinstance(duration, int) == True:
        mono_signal = mono_signal[0:sr*duration]

    return mono_signal, sr

"""
def spectrogram(signal, sr, num_frames, hop_length, num_freq, lowcut=500, highcut=15000):
    signal = pre_emphasis(signal)
    spectrum = librosa.core.stft(signal, n_fft=num_frames, hop_length=hop_length, win_length=None, window='hann', center=False)
    S = librosa.feature.melspectrogram(y=None, sr=sr, S=spectrum, n_fft=num_frames, hop_length=hop_length, n_mels=num_freq)
    return librosa.core.power_to_db(np.abs(S)**2).transpose()

def inverse_spectrogram(spectrogram, num_frames, hop_length):
    wave = librosa.core.istft(spectrogram, win_length=num_frames, hop_length=hop_length, window='hann', center=False)
    return wave
"""

def invert_spectrogram(spectrogram, fft_size, hop):
    x_reconstruct = audio_utilities.reconstruct_signal_griffin_lim(spectrogram,
                                                                   fft_size,
                                                                   hopsamp=hop,
                                                                   iterations=500)
    max_sample = np.max(abs(x_reconstruct))
    if max_sample > 1.0:
        x_reconstruct = x_reconstruct / max_sample
    return x_reconstruct
    
