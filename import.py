# Importamos las librerías básicas
import re
import numpy as np
import pandas as pd
import csv
import string # Para usar punctuation y eliminar los signos dee puntuación
import seaborn as sns
import matplotlib.pyplot as plt
import tensorflow
import ast # Para poder evaluar listas
from wordcloud import WordCloud # Para la nube de palabras
from nltk.util import ngrams
from nltk import FreqDist
from collections import Counter
from textblob import TextBlob

# Importamos SpaCy
import spacy
from spacy import displacy

# Importamos la función de train_text_split
from sklearn.model_selection import train_test_split

# Importamos los modelos que podemos necesitar.
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer

# Metricas
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

# Para usar redes neuronales recurrentes y convolucionales:
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.layers import Embedding, LSTM, Conv1D, MaxPooling1D, GlobalMaxPooling1D, Dense, Flatten
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam

# Cargamos el diccionario en inglés, que es el idioma de los comentarios que procesaremos.
pln = spacy.load('en_core_web_lg')
# Determinamos las stop-words por defecto
stop_words = pln.Defaults.stop_words

print (pln)
