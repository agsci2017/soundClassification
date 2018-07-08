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
	while len(arr)<220000:
		arr=np.concatenate([arr,arr])
		
	return arr[:220000]

def tofeat(f):
	y, sr = librosa.load(f)
	y, idx = librosa.effects.trim(y,top_db=36)
	y=multiply(y)
	S = librosa.feature.spectral_contrast(y=y, sr=sr)
	print(S.shape)
	r = S.flatten()
	r = (r-min(r))/(max(r)-min(r))
	return r

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
	
	row['prob'] = np.array(model.predict(row['bsa'].reshape((1,7,430 ,1)))).flatten().max()
	row['type'] = np.array(model.predict(row['bsa'].reshape((1,7,430 ,1)))).flatten().argmax()
	
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
