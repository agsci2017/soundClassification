# soundClassification
По этапам:

  I. chromaExtract.py - утилита для извлечения фич. Поместить в папку audio, содержащую аудиофайлы. Формирует файл snd_chromaA.pickle (внутри таблица в формате pandas).
  
  II. trainModel.py - утилита тренировки модели. Принимает файл snd_chromaA.pickle, лежащий в папке с утилитой. Производит model.json (структура) и model.h5 (веса).
