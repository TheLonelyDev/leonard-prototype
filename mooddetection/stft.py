import librosa
import numpy as np
import matplotlib.pyplot as plt
import librosa.display

y, sr = librosa.load(librosa.util.example_audio_file())
chroma_y = librosa.feature.chroma_stft(y=y, sr=sr)

S = librosa.stft(y, n_fft=4096)
chroma_s_abs = librosa.feature.chroma_stft(S=np.abs(S), sr=sr)

chroma_s = librosa.feature.chroma_stft(S=S, sr=sr)


plt.figure(figsize=(10, 4))
librosa.display.specshow(chroma_y, y_axis='chroma', x_axis='time')
plt.colorbar()
plt.title('Chromagram Y')
plt.tight_layout()
plt.show()

plt.figure(figsize=(10, 4))
librosa.display.specshow(chroma_s_abs, y_axis='chroma', x_axis='time')
plt.colorbar()
plt.title('Chromagram S ABS')
plt.tight_layout()
plt.show()


plt.figure(figsize=(10, 4))
librosa.display.specshow(chroma_s, y_axis='chroma', x_axis='time')
plt.colorbar()
plt.title('Chromagram S')
plt.tight_layout()
plt.show()



