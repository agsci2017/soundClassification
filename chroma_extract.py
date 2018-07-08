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
	
	while len(arr)<220000:
		arr=np.concatenate([arr,arr])
		
	return arr[:220000]


dt=dt.sample(frac=1)

i=0
for index, row in dt.iterrows():
	
	if True:

		y, sr = librosa.load(row['name'])
		
		y, idx = librosa.effects.trim(y,top_db=36)

		y=multiply(y)
		
		S = librosa.feature.spectral_contrast(y=y, sr=sr)
		
		print(S.shape)

		y=None
		
		r = S.flatten()
		r = (r-min(r))/(max(r)-min(r))

		row['bsa'] = r
		dt.iloc[index] = row
		S=None

		i+=1
		print(i/float(dt.count()[2]))
		

dt.to_pickle('snd_chromaA.pickle')
