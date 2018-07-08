# soundClassification
По этапам:

## Извлечение признаков

Поместить **chromaExtract.py** в папку **audio** и запустить. Сформирует файл **snd_chromaA.pickle** (внутри таблица в формате pandas, с именами файлов и признаками).

## Тренировка модели

Поместить **trainModel.py** в папку **audio**. Принимает файл **snd_chromaA.pickle**, лежащий в папке с утилитой. Производит **model.json** (структура) и **model.h5** (веса).

## Оценка (получение файла result.txt)

Поместить файл **estimateResult.py** в папку **test**. Туда же скопировать **model.json** и **model.h5**. Производит **result.txt**.
