import torch
from pydub import AudioSegment
import numpy as np


def read_audio_generic(input_file, monochannel=True, sampling_rate=16000, normalize=True):
    """
    Function to read files containing PCM (int16 or int32) audio
    """
    audiofile = AudioSegment.from_file(input_file, input_file.split('.')[-1])
    channels = audiofile.channels
    out_sampling_rate = audiofile.frame_rate
    if sampling_rate is None:
        sampling_rate = out_sampling_rate
    else:
        audiofile = audiofile.set_frame_rate(sampling_rate)
        assert audiofile.frame_rate == sampling_rate
        # print(out_sampling_rate, '->', sampling_rate)

    if monochannel and (channels > 1):
        audiofile = audiofile.set_channels(1)
        channels = audiofile.channels

    if audiofile.sample_width == 2:
        dtype = np.int16
        norm = 32768.0
    elif audiofile.sample_width == 4:
        dtype = np.int32
        norm = 2147483648
    else:
        assert False

    data = np.frombuffer(audiofile._data, dtype)

    if channels == 1:
        signal = data[:, None]
    elif data.size >= channels:
        signal = list()
        for chn in list(range(channels)):
            signal.append(data[chn::channels])
        signal = np.stack(signal, -1)
    else:
        signal = np.zeros((0, channels), dtype)

    if monochannel:
        signal = signal[:, 0]

    if normalize:
        signal = signal / norm
    return sampling_rate, signal

class ToSpecTorch(torch.nn.Module):

    def __init__(self, sampling_rate=16000, audio_norm_target_dBFS=-30, n_fft=512, window_step=10, window_length=25):
        super(ToSpecTorch, self).__init__()
        self.audio_norm_target = (10.0 ** (audio_norm_target_dBFS/20.0))
        self.hop_length = int(sampling_rate * window_step / 1000)
        win_length = int(sampling_rate * window_length / 1000)
        self.win = int(max(win_length, n_fft))
        from torchaudio.transforms import Spectrogram
        self.op = Spectrogram(
            n_fft=int(n_fft),
            win_length=win_length,
            hop_length=self.hop_length,
            power=1, pad=0, normalized=False,
            center=False, onesided=True)

    def get_len(self, samples):
         return samples * self.hop_length + self.win - self.hop_length

    def get_stride(self, samples):
        return samples * self.hop_length

    def __call__(self, wav):
        wave_std = torch.sqrt(torch.mean(wav ** 2))
        factor = self.audio_norm_target / wave_std
        if factor > 1.0:
            wav = wav * factor

        spec = self.op(wav)
        return spec


class TestDataset:

    def __init__(self, list_all, audio_len, audio_stride):
        self.list_all = list_all
        self.audio_len = audio_len
        self.audio_stride = audio_stride

    def __len__(self):
        return len(self.list_all)

    def __getitem__(self, index):
        dat = self.list_all[index]
        filewav = dat[1]
        audio = np.float32(read_audio_generic(filewav)[1])

        if len(audio) < self.audio_len:
            # print('to short:', len(audio),  self.audio_len)
            audio = np.pad(audio, (0, self.audio_len - len(audio)), 'wrap')
            # return None

        audio = torch.from_numpy(audio).unfold(0, self.audio_len, self.audio_stride)  # B x T
        assert audio.shape[1] == self.audio_len
        label = [dat[0], ] * len(audio)
        return label, audio