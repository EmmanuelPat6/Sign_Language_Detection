import pickle
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import numpy as np

#Import dei dati dal file data.pickle creato con create_dataset.py
#Data Dictionary
data_dict = pickle.load(open('./data.pickle', 'rb'))

#Verifica delle lunghezze dei dati per garantire che siano omogenee
data_lengths = [len(d) for d in data_dict['data']]
print("Lunghezze dei dati:", set(data_lengths))  # Stampa le lunghezze dei dati per verificarle

#Trova la lunghezza più comune o la massima lunghezza
expected_length = max(set(data_lengths), key=data_lengths.count)  #Lunghezza più comune

#Uniforma i dati: scarta i dati che non hanno la lunghezza corretta
data_cleaned = [d for d in data_dict['data'] if len(d) == expected_length]
labels_cleaned = [l for i, l in enumerate(data_dict['labels']) if len(data_dict['data'][i]) == expected_length]

#Converte i dati e le etichette in array numpy
data = np.asarray(data_cleaned)
labels = np.asarray(labels_cleaned)

#Suddivisione del dataset in training e test
x_train, x_test, y_train, y_test = train_test_split(data, labels, test_size=0.2, shuffle=True, stratify=labels)

#Creazione e addestramento del modello
model = RandomForestClassifier()
model.fit(x_train, y_train)

#Predizione e valutazione delle performance
y_predict = model.predict(x_test)
score = accuracy_score(y_predict, y_test)

#Stampa del risultato
print('{}% of samples were classified correctly !'.format(score * 100))

#Salvataggio del modello addestrato
f = open('model.p', 'wb')
pickle.dump({'model': model}, f)
f.close()
