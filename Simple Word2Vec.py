# Wikipedia Models:
# https://wikipedia2vec.github.io/wikipedia2vec/pretrained/

from gensim.models import KeyedVectors
import pickle

# Laden des vortrainierten word2vec-Modells im .pkl-Format
with open('path/to/word2vec/dewiki_20180420_100d.pkl', 'rb') as file:
    model = pickle.load(file)

# Zwei Wörter, deren Ähnlichkeit wir überprüfen möchten
wort1 = 'Auto'
wort2 = 'Fahrzeug'

# Berechnung der Ähnlichkeit zwischen den beiden Wörtern
aehnlichkeit = model.similarity(wort1, wort2)

print(f'Die Ähnlichkeit zwischen "{wort1}" und "{wort2}" beträgt: {aehnlichkeit}')

