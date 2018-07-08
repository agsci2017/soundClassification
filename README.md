# soundClassification
По этапам:
  I. chromaExtract.py - утилита для извлечения фич из файлов в папке audio. Формирует файл snd_chromaA.pickle .
  
  II. trainModel.py - утилита тренировки модели. Принимает на вход snd_chromaA.pickle, лежащий в папке с утилитой. Производит model.json (структура) и model.h5 (веса).
