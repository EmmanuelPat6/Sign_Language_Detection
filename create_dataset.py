import os
import pickle
import mediapipe as mp
import cv2
import matplotlib.pyplot as plt #Per verificare se il codice funziona plottando un'immagine per ogni classe

#Itera su tutte le immagini per ciascuna classe estraendo da ciascuna di esse i punti di riferimento (LANDMARKS).
#Successivamente tutti questi dati verranno salvati in un unico file che verrà usato per l'addestramento del CLASSIFICATORE

#Queste tre componenti sono utilizzate per detectare tutti i LANDMARKS (21 punti) in maniera tale da mostrare a video
#la posizione della mano e delle dita attraverso il posizionamento di ogni punto
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

#Hand Detector
#Definizione di un oggetto Hands
hands = mp_hands.Hands(static_image_mode=True, min_detection_confidence=0.3)

#Data Directory
DATA_DIR = './data'

#data e labels sono due variabile che conterranno tutte le informazioni necessarie alla detection delle varie lettere
#labels fa riferimento alle classi
#data contiene tutti i dati delle immagine ottenute precedentemente
data = []
labels = []

#Processo Iterativo
for dir_ in os.listdir(DATA_DIR):
    for img_path in os.listdir(os.path.join(DATA_DIR, dir_)):   #se aggiungo [:1]: plotto solo la prima immagine per ogni classe (utile per eventuali prove)
        data_aux = []

        x_ = []
        y_ = []

        #Conversione delle immagini in RGB per inserirle in MediaPipe
        img = cv2.imread(os.path.join(DATA_DIR, dir_, img_path))
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        #Detection di tutti i LANDMARKS nell'immagine e iterazione per tutti quelli ottenuti
        results = hands.process(img_rgb)
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:


                # DEBUG: plot dei LANDMARKS per una sola immagine
                # mp_drawing.draw_landmarks(img_rgb, hand_landmarks, mp_hands.HAND_CONNECTIONS,
                #                           mp_drawing_styles.get_default_hand_landmarks_style(),
                #                           mp_drawing_styles.get_default_hand_connections_style())

                # plt.figure()
                # plt.imshow(img_rgb)
                # plt.show()

                #Creazione di un array con tutti i LANDMARKS
                #Prende tutte le immagini e da ognuna di esse prende le informazioni date dalla detection dei LANDMARKS
                for i in range(len(hand_landmarks.landmark)):
                    x = hand_landmarks.landmark[i].x
                    y = hand_landmarks.landmark[i].y

                    x_.append(x)
                    y_.append(y)

                for i in range(len(hand_landmarks.landmark)):
                    x = hand_landmarks.landmark[i].x
                    y = hand_landmarks.landmark[i].y
                    data_aux.append(x - min(x_))
                    data_aux.append(y - min(y_))

            data.append(data_aux)
            labels.append(dir_)

#Salvataggio di queste informazioni all'interno del file data.pickle
f = open('data.pickle', 'wb')
pickle.dump({'data': data, 'labels': labels}, f)
f.close()
