from __future__ import print_function
import keras
import keras.datasets
from keras.models import Sequential, Model
from keras.models import model_from_json
import random
import pickle
import pandas as pd
import numpy as np
import sys
import glob
from sklearn.model_selection import train_test_split
import librosa
import librosa.display
import matplotlib.pyplot as plt

model = model_from_json(open('model.json', 'r').read())
model.load_weights("model.h5")
print("Loaded model from disk")

def multiply(arr):
	while len(arr)<250000:
		arr=np.concatenate([arr,arr])
		
	return arr[:250000]

def tofeat(f):
	y, sr = librosa.load(f)
	#y, idx = librosa.effects.trim(y,top_db=36)
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
	
	return SR.flatten()

samples = []
samples.append((0,"*.wav"))


names = []
types = []
bsas = []

for s in samples:
	print(s[0], len(glob.glob(s[1])))
	for f in glob.glob(s[1]):
		names.append(f)
		types.append(s[0])


dt = pd.DataFrame({'name': names, 'type': types, 'bsa': None, 'prob': None})

print(dt.head(5))


for index, row in dt.iterrows():
	
	row['bsa'] = tofeat(row['name'])
	
	row['prob'] = np.array(model.predict(row['bsa'].reshape((1,12,489 ,1)))).flatten().max()
	row['type'] = np.array(model.predict(row['bsa'].reshape((1,12,489 ,1)))).flatten().argmax()
	
	dt.iloc[index] = row
	
	print(row)

f = open("result.txt","w")

def tran(x):
	if x==0: return "background"
	if x==1: return "bags"
	if x==2: return "door"
	if x==3: return "keyboard"
	if x==4: return "ring"
	if x==5: return "knocking_door"
	if x==6: return "speech"
	if x==7: return "tool"

for index, row in dt.iterrows():
	f.write("""{} {} {}\n""".format(row['name'],row['prob'],tran(row['type'])))

f.close()

sys.exit(0)
