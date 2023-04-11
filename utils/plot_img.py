import numpy as np
import cv2


if __name__ == '__main__':
    import librosa
    audio_path = '123.mp3'
    note_path = '123_3.txt'
    notes = []
    with open(note_path) as f:
        for line in f:
            line = line.strip().split()[: 3]
            onset, offset, pitch = [float(x) for x in line]
            notes.append([onset, offset, pitch, 1.0])
    cqt_config = {'hop': 512, 'n_bins': 176, 'bins_per_octave': 24, 'sr': 16000}
    audios, _ = librosa.load(audio_path, sr=cqt_config['sr'], mono=True, dtype='float32')
    cqts = []
    length = 3 * cqt_config['sr']
    n_segments = int(np.ceil(len(audios) / length))
    for i in range(n_segments):
        audio = audios[i*length : (i+1)*length]
        cqt = librosa.cqt(audio, sr=cqt_config['sr'], hop_length=cqt_config['hop'], fmin=27.5, n_bins=cqt_config['n_bins'],
                        bins_per_octave=cqt_config['bins_per_octave'])
        cqt = librosa.amplitude_to_db(np.abs(cqt))
        cqts.append(cqt)
    cqt = np.concatenate(cqts, 1)

    cqt = librosa.cqt(audios, sr=cqt_config['sr'], hop_length=cqt_config['hop'], fmin=27.5, n_bins=cqt_config['n_bins'],
                        bins_per_octave=cqt_config['bins_per_octave'])
    cqt = librosa.amplitude_to_db(np.abs(cqt))

    dumppath = '123_3.png'

    height, width = cqt.shape
    resize_width, resize_height = width * 2, height * 2
    feature = cv2.resize(cqt, (resize_width, resize_height))
    feature = feature - feature.min()
    feature /= (feature.max() + 1e-6)
    feature = (feature * 255).astype('uint8')
    img = cv2.applyColorMap(feature, cv2.COLORMAP_VIRIDIS)
    time_length = width * cqt_config['hop'] / cqt_config['sr']
    n_bins = cqt_config['n_bins']

    for index, (onset, offset, pitch, conf) in enumerate(notes):
        x1 = int(onset * resize_width / time_length)
        x2 = int(offset * resize_width / time_length)
        y1 = int((pitch - 21) * cqt_config['bins_per_octave'] / 12 / n_bins * resize_height)
        y2 = resize_height - 1
        img = cv2.rectangle(img, (x1, y1), (x2, y2), (0, 0, 255))
        img = cv2.putText(img, '{:.2f}'.format(conf), (x1, y1 - 5 if index % 2 == 0 else y1 + 12), 0, 0.4, (0, 0, 255))

    cv2.imwrite(dumppath, img)

    