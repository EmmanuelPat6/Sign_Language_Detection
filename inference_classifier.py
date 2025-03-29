import pickle
import cv2
import mediapipe as mp
import numpy as np

model_dict = pickle.load(open('./model.p', 'rb'))
model = model_dict['model']

cap = cv2.VideoCapture(0)

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

hands = mp_hands.Hands(static_image_mode=True, min_detection_confidence=0.5)

labels_dict = {
    0: 'A', 1: 'B', 2: 'C', 3: 'D', 4: 'E', 5: 'F', 6: 'G', 7: 'I', 8: 'L',
    9: 'M', 10: 'N', 11: 'O', 12: 'R', 13: 'S', 14: 'T', 15: 'U', 16: 'V', 17: 'Y'
}
Letters = ""

while True:

    data_aux = []
    x_ = []
    y_ = []

    ret, frame = cap.read()

    H, W, _ = frame.shape

    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    results = hands.process(frame_rgb)

    if results.multi_hand_landmarks:
        # Consideriamo solo la prima mano rilevata
        hand_landmarks = results.multi_hand_landmarks[0]  # Prendi solo la prima mano
        mp_drawing.draw_landmarks(
            frame,  # image to draw
            hand_landmarks,  # model output
            mp_hands.HAND_CONNECTIONS,  # hand connections
            mp_drawing_styles.get_default_hand_landmarks_style(),
            mp_drawing_styles.get_default_hand_connections_style())

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

        # Gestiamo l'area del bounding box
        if x_ and y_:
            x1 = int(min(x_) * W) - 10
            y1 = int(min(y_) * H) - 10
            x2 = int(max(x_) * W) - 10
            y2 = int(max(y_) * H) - 10

        try:
            # Predizione del modello
            prediction = model.predict([np.asarray(data_aux)])
            predicted_character = labels_dict[int(prediction[0])]
        except Exception as e:
            print("Errore nella predizione:", e)
            predicted_character = "?"  # Usa un carattere di default in caso di errore

        print(predicted_character)

        Letters = Letters + " " + predicted_character

        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 0), 4)
        cv2.putText(frame, predicted_character, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 1.3, (0, 0, 0), 3,
                    cv2.LINE_AA)

    cv2.imshow('frame', frame)
    cv2.waitKey(1)

    cv2.putText(frame, 'If you want to quit press Q!', (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.3, (255, 255, 0), 2,
                cv2.LINE_AA)
    cv2.imshow('frame', frame)

    if cv2.waitKey(100) == ord('q'):
        print("Letters", Letters)
        f = open('predicted_character.txt', 'w')
        f.write(Letters)
        f.close()
        cv2.destroyAllWindows()
        break
