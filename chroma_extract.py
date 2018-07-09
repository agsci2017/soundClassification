import glob
import pandas as pd
import numpy as np
import librosa
import librosa.display
import matplotlib.pyplot as plt


samples = []
samples.append((0,'background_*.wav'))
samples.append((1,'bags_*.wav'))
samples.append((1,'bg_*.wav'))
samples.append((1,'t_bags_*.wav'))


samples.append((2,'door_*.wav'))
samples.append((2,'d_*.wav'))
samples.append((2,'t_door_*.wav'))

samples.append((3,'keyboard_*.wav'))
samples.append((3,'k_*.wav'))
samples.append((3,'t_keyboard_*.wav'))
samples.append((3,'tt_k_*.wav'))

samples.append((4,'ring_*.wav'))
samples.append((4,'t_ring_*.wav'))

samples.append((5,'t_knocking_door_*.wav'))
samples.append((5,'knocking_*.wav'))
samples.append((5,'tt_kd_*.wav'))

samples.append((6,'speech_*.wav'))

samples.append((7,'tool_*.wav'))

names = []
types = []

for s in samples:
	print(s[0], len(glob.glob(s[1])))
	for f in glob.glob(s[1]):
		names.append(f)
		types.append(s[0])


dt = pd.DataFrame({'name': names, 'type': types, 'bsa': None})

print(dt.head(5))


def multiply(arr):
	
	while len(arr)<250000:
		arr=np.concatenate([arr,arr])
		
	return arr[:250000]


dt=dt.sample(frac=1)

i=0
for index, row in dt.iterrows():
	
	if True:

		y, sr = librosa.load(row['name'])
		
		#y, idx = librosa.effects.trim(y,top_db=40)

		y=multiply(y)
		
		S1 = librosa.feature.spectral_contrast(y=y, sr=sr)
		from sklearn.preprocessing import scale
		S1 = scale( S1 )
		
		#print(S1.shape)
		S2 = librosa.feature.spectral_rolloff(y=y, sr=sr)
		#print(S2.shape)
		S2 = scale( S2 ,axis=1)
		
		S3 = librosa.feature.spectral_flatness(y=y)
		#print(S3.shape)
		S3 = scale( S3 ,axis=1)
		
		S4 = librosa.feature.spectral_bandwidth(y=y, sr=sr)
		S4 = scale( S4 ,axis=1)
		
		S5 = librosa.feature.spectral_centroid(y=y, sr=sr)
		S5 = scale( S5 ,axis=1)
		
		S6 = librosa.feature.rmse(y=y)
		S6 = scale( S6 ,axis=1)
		
		SR = np.vstack([S5,S4,S2,S3,S1,S6])
		
		print(SR.shape)
		#~ librosa.display.specshow(SR)
		#~ plt.show()
		
		#librosa.display.specshow(SR)
		#plt.show()
		y=None
		
		row['bsa'] = SR.flatten()
		dt.iloc[index] = row
		
		SR=None
		S1=None
		S2=None
		S3=None
		S4=None
		S5=None
		S6=None

		i+=1
		print(i/float(dt.count()[2]))
	#break

dt.to_pickle('snd_chromaA.pickle')
