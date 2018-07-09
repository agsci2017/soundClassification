# soundClassification

Используется фреймворк Keras (для языка программирования Python).

Установка необходимых модулей:

`sudo pip3 install keras`
`sudo pip3 install librosa`
`sudo pip3 install pandas`
`sudo pip3 install numpy`
`sudo pip3 install scikit-learn`

По этапам:

## Извлечение признаков

Поместить **chromaExtract.py** в папку **audio** и запустить. На основе аудиофайлов, сформирует файл **snd_chromaC.pickle** (внутри таблица в формате pandas, с именами файлов и признаками).

## Тренировка модели

Поместить **trainModel.py** в папку **audio**. Принимает файл **snd_chromaC.pickle**, лежащий в папке с утилитой. Производит **model.json** (структура) и **model.h5** (веса).

## Оценка (получение файла result.txt)

Поместить файл **estimateResult.py** в папку **test**. Туда же скопировать **model.json** и **model.h5**. Производит **result.txt**.
