import numpy as np
import soundfile
from numpy.lib.stride_tricks import as_strided

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
        timesteps[first : last] = 1

    return timesteps

def mark_spectro(spectro_length, num_frames,
                 hop_length, markers, threshold=0.5):
    """
    function that calculates the overlap from the previously computed markers
    """
    overlaps = np.array([markers[i : (i + num_frames)].mean()
                for i in range(0, len(markers), hop_length)])
        
    overlaps[overlaps >= threshold] = 1
    overlaps[overlaps < threshold] = 0
    
    overlaps = overlaps[:spectro_length]

    return overlaps


def load_file(filename):
    with soundfile.SoundFile(filename) as sound_file:
        audio = sound_file.read(dtype='float32')
        sample_rate = sound_file.samplerate

    # convert to mono if necessary:
    if audio.ndim >= 2:
        audio = np.mean(audio, 1)
    return audio, sample_rate


def spectrogram(samples, fft_length=256, sample_rate=2, hop_length=128):
    """
    Compute the spectrogram for a real signal.
    The parameters follow the naming convention of
    matplotlib.mlab.specgram

    Args:
        samples (1D array): input audio signal
        fft_length (int): number of elements in fft window
        sample_rate (scalar): sample rate
        hop_length (int): hop length (relative offset between neighboring
            fft windows).

    Returns:
        x (2D array): spectrogram [frequency x time]
        freq (1D array): frequency of each row in x

    Note:
        This is a truncating computation e.g. if fft_length=10,
        hop_length=5 and the signal has 23 elements, then the
        last 3 elements will be truncated.
    """
    assert not np.iscomplexobj(samples), "Must not pass in complex numbers"

    window = np.hanning(fft_length)[:, None]
    window_norm = np.sum(window**2)

    # The scaling below follows the convention of
    # matplotlib.mlab.specgram which is the same as
    # matlabs specgram.
    scale = window_norm * sample_rate

    trunc = (len(samples) - fft_length) % hop_length
    x = samples[:len(samples) - trunc]

    # "stride trick" reshape to include overlap
    nshape = (fft_length, (len(x) - fft_length) // hop_length + 1)
    nstrides = (x.strides[0], x.strides[0] * hop_length)
    x = as_strided(x, shape=nshape, strides=nstrides)

    # window stride sanity check
    assert np.all(x[:, 1] == samples[hop_length:(hop_length + fft_length)])

    # broadcast window, compute fft over columns and square mod
    x = np.fft.rfft(x * window, axis=0)
    x = np.absolute(x)**2

    # scale, 2.0 for everything except dc and fft_length/2
    x[1:-1, :] *= (2.0 / scale)
    x[(0, -1), :] /= scale

    freqs = float(sample_rate) / fft_length * np.arange(x.shape[0])

    return x, freqs


def build_spectrogram(audio, sample_rate, window=20, step=10, max_freq=None, eps=1e-14):
    """ Calculate the log of linear spectrogram from FFT energy
    Params:
        filename (str): Path to the audio file
        step (int): Step size in milliseconds between windows
        window (int): FFT window size in milliseconds
        max_freq (int): Only FFT bins corresponding to frequencies between
            [0, max_freq] are returned
        eps (float): Small value to ensure numerical stability (for ln(x))
    """
    if max_freq is None:
        max_freq = sample_rate / 2
    if max_freq > sample_rate / 2:
        raise ValueError("max_freq must not be greater than half of sample rate")
    if step > window:
        raise ValueError("step size must not be greater than window size")

    hop_length = int(0.001 * step * sample_rate)
    fft_length = int(0.001 * window * sample_rate)
    pxx, freqs = spectrogram(audio, fft_length=fft_length,
                             sample_rate=sample_rate, hop_length=hop_length)
    ind = np.where(freqs <= max_freq)[0][-1] + 1
    return np.transpose(np.log(pxx[:ind, :] + eps))