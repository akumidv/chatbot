
echo "Чтобы вытащить файлы из докера docker run -it chatbot --rm --name ruchatbot bash -c \"/chatbot/scripts/console_bot.sh\" и в другой коносли docker cp  ruchatbot:chatbot .""
echo "Предварительно надо скачать ruword2tags.db. Актуальная ссылка в репозитории https://github.com/Koziev/ruword2tags"
mkdir temp
cd temp
git clone https://github.com/Koziev/ruword2tags
copy ..\ruword2tags.db .\ruword2tags\ruword2tags.db /y
cd ruword2tags
pip install .
cd ..


rem keras==2.4.3 tensorflow==2.3.1
pip install tensorflow --use-feature=2020-resolver
pip install sentencepiece lightgbm scikit-learn==0.24.0 gensim pathlib python-crfsuite colorama coloredlogs requests flask flask_sqlalchemy flask_wtf h5py pyconll ufal.udpipe pyyaml --use-feature=2020-resolver
pip install tensorflow-addons --use-feature=2020-resolver
pip install python-Levenshtein
pip install transformers
pip install torch
rem pip install git+https://www.github.com/keras-team/keras-contrib.git

pip install python-telegram-bot --upgrade

pip install git+https://github.com/Koziev/rulemma
pip install git+https://github.com/Koziev/rutokenizer
pip install git+https://github.com/Koziev/rupostagger
rem #RUN pip install git+https://github.com/Koziev/ruword2tags
pip install git+https://github.com/Koziev/rusyllab
pip install git+https://github.com/Koziev/ruchunker



cd ..
