'''
数据增强方法，包括时域增强和频域数据增强
https://www.cnblogs.com/LXP-Never/p/13404523.html
https://github.com/pyyush/SpecAugment
'''
import numpy as np
import librosa
import soundfile as sf
import unittest
import os
import io
from subprocess import Popen, PIPE
import sys

# 时域增强方法在cpu进行，频域增强在gpu上进行
class Augmentor(object):
  def __init__(self, sr, bins_per_octave):
    self.sr = sr
    self.bins_per_octave = bins_per_octave

  def snr2noise(self, clean, noise, SNR):
    p_clean = np.mean(clean ** 2)
    p_noise = np.mean(noise ** 2)
    scalar = np.sqrt(p_clean / (10 ** (SNR / 10)) / (p_noise + np.finfo(np.float32).eps))
    noisy = clean + scalar * noise
    return noisy, scalar

  # 绝对音量
  def volumeAu(self, wav, dB):
    power = np.mean(wav ** 2)
    scalar = np.sqrt(10 ** (dB / 10) / (power + np.finfo(np.float32).eps))
    auwav = wav * scalar
    return auwav, scalar

  def time_shift(self, wav, time_shift):
    return np.roll(wav, int(time_shift * self.sr))

  # 谐波失真 会改变音量
  def harmonic_distortion(self, wav, COUNT):
    hwav = 2 * np.pi * wav
    count = 0
    while count < COUNT:
        hwav = np.sin(hwav)
        count += 1
    return hwav

  def pitch_shift(self, wav, midi_shift=1):
    # 一个半音对应的midi值为1  n_steps代表偏移多少个频点
    n_steps = midi_shift * (self.bins_per_octave // 12)
    pwav = librosa.effects.pitch_shift(wav, sr=self.sr, n_steps=n_steps, 
                                       bins_per_octave=self.bins_per_octave)
    return pwav

  def _memtmpfs_adjust_speed_wav(self, wav, speed):

    byte_io = io.BytesIO()
    sf.write(byte_io, wav, self.sr, subtype='FLOAT', format='WAV')
    byte_io.seek(0)
    wav = byte_io.read()
    # with open('testin.wav', 'wb') as f:
    #   f.write(wav)
  
    p = Popen("ffmpeg -loglevel quiet -y -f wav -ac 1 -channel_layout mono -i - -filter_complex \
              'atempo=tempo={:.2f}' -f wav -ar {} -ac 1 -".format(speed, self.sr),
               stdin=PIPE, stdout=PIPE, shell=True)
    out = p.communicate(wav)[0]
    # with open('testout.wav', 'wb') as f:
    #   f.write(out)
    wav, _ = sf.read(io.BytesIO(out), dtype='float32')
    return wav

  def _memtmpfs_adjust_speed_raw(self, wav, speed):

    byte_io = io.BytesIO()
    sf.write(byte_io, wav, self.sr, subtype='FLOAT', format='RAW')
    byte_io.seek(0)
    wav = byte_io.read()
    # sf.write('testin.wav', np.frombuffer(wav, dtype='float32'), self.sr)
    p = Popen("ffmpeg -loglevel quiet -y -f f32le -ar {} -ac 1 -channel_layout mono -i - -filter_complex \
              'atempo=tempo={:.2f}' -f f32le -ar {} -".format(self.sr, speed, self.sr),
               stdin=PIPE, stdout=PIPE, shell=True)
    wav = p.communicate(wav)[0]
    # wav, _ = sf.read(io.BytesIO(wav), dtype='float32', samplerate=self.sr,
    #                  channels=1, subtype='FLOAT', format='RAW')

    wav = np.frombuffer(wav, dtype='float32')
    # sf.write('testout.wav', wav, self.sr)
    
    return wav
  
  # 变速不变调
  def adjust_speed(self, wav, speed):
    value = self._memtmpfs_adjust_speed_wav(wav, speed)
    return value


class TestAugmentor(unittest.TestCase):
  def setUp(self):
    self.sr = 16000
    self.augmentor = Augmentor(sr=self.sr, bins_per_octave=48)
    self.wav, sr = sf.read('/home/data/SSVD-v2.0/test16k/100135.wav')
    if sr != self.sr:
      self.wav = librosa.resample(self.wav, orig_sr=sr, target_sr=self.sr)
    self.noise, sr = sf.read('/home/data/wxk/TUTDataset/TUT-acoustic-scenes-2017-development/home/a029_0_10.wav', always_2d=True)
    self.noise = self.noise.T[0]
    if sr != self.sr:
      self.noise = librosa.resample(self.noise, orig_sr=sr, target_sr=self.sr)
    min_length = min(len(self.wav), len(self.noise))
    self.wav = self.wav[: min_length]
    self.noise = self.noise[: min_length]

  # 信噪比[5, 15] 需要尝试
  def test_snr2noise(self):
    SNR = 10
    noisy, _ = self.augmentor.snr2noise(self.wav, self.noise, SNR)
    if noisy.max() > 1.0:
      factor = 1. / (noisy.max() + 1e-7)
      noisy *= factor
      wav = self.wav * factor
    noise = noisy - wav

    os.makedirs('snr2noise', exist_ok=True)
    sf.write('snr2noise/noisy10.wav', noisy, self.sr)
    sf.write('snr2noise/clean10.wav', wav, self.sr)
    sf.write('snr2noise/noise10.wav', noise, self.sr)

  # 可以设置volume [0.2, 1] 需要测试
  def test_volumeAu(self):
    # dB = -13
    dB = -24
    auwav, _ = self.augmentor.volumeAu(self.wav, dB)
    if auwav.max() > 1.0:
      factor = 1. / (auwav.max() + 1e-7)
      auwav *= factor
      wav, _ = self.augmentor.volumeAu(auwav, -dB)
    os.makedirs('volumeAu', exist_ok=True)

    sf.write('volumeAu/wav.wav', self.wav, self.sr)
    sf.write('volumeAu/auwav.wav', auwav, self.sr)
  
  # test_time_shift [-1, 1]
  def test_time_shift(self):
    time_shift = 0.2
    wav = self.augmentor.time_shift(self.wav, time_shift)
    os.makedirs('timeshift', exist_ok=True)

    sf.write('timeshift/wav.wav', self.wav, self.sr)
    sf.write('timeshift/shiftwav.wav', wav, self.sr)
  
  # COUNT可以选择[1, 3]
  def test_harmonic_distortion(self):
    COUNT = 5
    wav = self.augmentor.harmonic_distortion(self.wav, COUNT)
    wav /= (wav.max() + 1e-7)
    os.makedirs('harmonic_distortion', exist_ok=True)

    sf.write('harmonic_distortion/wav.wav', self.wav, self.sr)
    sf.write('harmonic_distortion/harmonic_distortion.wav', wav, self.sr)

  # 音高偏移可以选择半音[-2, 2]
  def test_pitch_shift(self):
    midi_shift = 2
    wav = self.augmentor.pitch_shift(self.wav, midi_shift)
    os.makedirs('pitch_shift', exist_ok=True)

    sf.write('pitch_shift/wav.wav', self.wav, self.sr)
    sf.write('pitch_shift/pitch_shift.wav', wav, self.sr)

  # 速度可以选择[0.5, 2]
  def test_adjust_speed(self):
    speed = 2
    wav = self.augmentor.adjust_speed(self.wav, speed)
    os.makedirs('adjust_speed', exist_ok=True)

    sf.write('adjust_speed/wav.wav', self.wav, self.sr)
    sf.write('adjust_speed/adjust_speed.wav', wav, self.sr)


if __name__ == '__main__':
  unittest.main()
