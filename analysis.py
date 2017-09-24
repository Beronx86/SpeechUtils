import librosa, numpy as np, scipy.io.wavfile as sio_wav, json
from scipy import signal

def load_wav(file_path, fs) -> np.ndarray:
    """
    Load a wav file.
    :param file_path: the path of the wav file
    :param fs: sampling frequency (Hz)
    :return: array, [-1, 1]
    """
    assert file_path[-3:] == "wav", "[!] Only .wav file can be read.\n{}".format(file_path)
    wav, _ = librosa.core.load(file_path, fs)
    assert isinstance(wav, np.ndarray)
    return wav

def save_wav(file_path, wav, fs, norm=True):
    """
    Save a wav to destination[file_path].
    :param file_path: the path of the wav file
    :param wav: np_array, shape := (time_step, ...)
    :param fs: sampling frequency (Hz)
    :param norm: if True [default], the wav is normalized to [-1, 1].
    :return: None
    """
    assert file_path[-3:] == "wav", "[!] Only .wav type is supported.\n{}".format(file_path)
    if norm:
        wav /= np.max(np.abs(wav))
    sio_wav.write(file_path, fs, wav.astype("float32"))

def load_from_json(file_path, tool_cls):
    """
    Build analysis object from json file.
    :param file_path: json file path, which contains the meta information of a analysis tool
    :param tool_cls: tool class
    :return: a tool class object
    """
    with open(file_path, "r") as f:
        meta = json.load(f)
    return tool_cls(**meta)

def pre_emphasis(wav, alpha=0.97):
    return signal.lfilter([1, -alpha], [1], wav)

def de_emphasis(wav, alpha=0.97):
    return signal.lfilter([1], [1, -alpha], wav)

def get_stft_mag(wav, n_fft, frame_shift_dots, frame_length_dots, window_type="hann", **kargs):
    tmp = np.abs(librosa.core.stft(wav, n_fft, frame_shift_dots, frame_length_dots, window_type))
    return tmp.T

def get_mel(stft_m, n_mels=80):
    tmp = librosa.feature.melspectrogram(S=np.square(stft_m.T), n_mels=n_mels)
    return tmp.T

class AnalysisToolBase(object):
    def __init__(self, fs, frame_shift, frame_length, n_fft, window_type):
        frame_shift_dots = int(frame_shift * fs)
        frame_length_dots = int(frame_length * fs)
        self.__meta = dict(fs=fs, frame_shift=frame_shift, frame_length=frame_length,
                           frame_shift_dots=frame_shift_dots, frame_length_dots=frame_length_dots,
                           n_fft=n_fft, window_type=window_type)

    @property
    def meta(self):
        return self.__meta

    def save_as_json(self, file_path):
        with open(file_path, "w") as f:
            json.dump(self.meta, f)

class GLA(AnalysisToolBase):
    """Griffin-Lim Vocoder
    """
    def __init__(self, *args, n_mels=80, pre_emphasis_coef=0.97):
        """
        :param args:
        :param pre_emphasis_coef:
        """
        super(GLA, self).__init__(*args)
        self.__meta['n_mels'] = n_mels
        self.__meta['pre_emphasis'] = pre_emphasis_coef
        self.__mel_filter = librosa.filters.mel(sr=self.meta.get('fs'),
                                                n_fft=self.meta.get('n_fft'), n_mels=self.meta.get('n_mels'))
        self.__mel_filter = np.transpose(self.__mel_filter, (1, 0))

    def extract(self, file_path):
        """
        Extract spectrogram and mel_spectrogram.
        :param file_path: wav file path
        :return: a list := [mel_spectrogram, spectrogram]
        """
        wav = pre_emphasis(load_wav(file_path, self.meta.get('fs')), self.meta.get('pre_emphasis'))
        stft_m = get_stft_mag(wav, **self.meta)
        mel = np.matmul(stft_m, self.__mel_filter)
        return [mel, stft_m]

    def synthesis(self, stft_m, aug_by_power=1.2, max_iter=50, norm=True):
        """
        Synthesize wave from spectrogram
        :param stft_m: spectrogram, shape := (time_step, ...)
        :param aug_by_power:
        :param max_iter:
        :param norm:
        :return:
        """
        aug_stft_m = np.power(stft_m.T, aug_by_power)
        wav_dots = (len(aug_stft_m) - 1) * self.meta.get('frame_shift_dots')
        wav = np.random.uniform(low=-1., high=1., size=(wav_dots,))
        for idx in range(max_iter):
            spec_complex =self.__stft(wav)
            spec_complex = aug_stft_m * spec_complex / np.abs(spec_complex)
            wav = self.__inv_stft(spec_complex)
        wav = de_emphasis(wav, self.meta.get('pre_emphasis'))
        if norm:
            wav /= np.max(np.abs(wav))
        return wav

    def __stft(self, y):
        return librosa.core.stft(y, n_fft=self.meta.get('n_fft'),
                                  hop_length=self.meta.get('frame_shift_dots'),
                                  win_length=self.meta.get('frame_length_dots'),
                                  window=self.meta.get('window_type'))

    def __inv_stft(self, spec):
        return librosa.core.istft(spec, window=self.meta.get('window_type'),
                                   hop_length=self.meta.get('frame_shift_dots'),
                                   win_length=self.meta.get('frame_length_dots'))
