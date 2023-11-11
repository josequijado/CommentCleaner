chmod +x *.bat
source odio_env/bin/activate
pip install -U pip setuptools wheel
pip install -U spacy
pip install seaborn
pip install scikit-learn
pip install sklearn
pip install tensorflow
pip install huggingface
pip install spacy-transformers
pip install wordcloud
pip install nltk
pip install textblob
python -m spacy download en_core_web_lg
huggingface-cli install spacy/en_core_web_lg_trf
