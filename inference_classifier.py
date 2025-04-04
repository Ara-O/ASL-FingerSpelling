import cv2
import mediapipe as mp
import pickle
import numpy as np

cap = cv2.VideoCapture(0)

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

hands = mp_hands.Hands(static_image_mode=True, min_detection_confidence=0.3)
model_dict = pickle.load(open('model.p', 'rb'))

model = model_dict['model']

labels_dict = {
0: 'A', 1: 'B', 2: 'C', 3: 'D', 4: 'E', 5: 'F', 6: 'G', 7: 'H', 8: 'I', 9: 'K', 10: 'L', 11: 'M', 12: 'N', 13: 'O', 14: 'P', 15: 'Q', 16: 'R', 17: 'S', 18: 'T', 19: 'U', 20: 'V', 21: 'W', 22: 'X', 23: 'Y', 24: 'J', 25: 'Z',
'I Love You': "I Love You"
#0: 'A', 1: 'B', 2: 'C', 3: 'D', 4: 'E', 5: 'F', 6: 'G', 7: 'H', 8: 'I', 9: 'J', 10: 'K', 11: 'L', 12: 'M', 13: 'N', 14: 'O', 15: 'P', 16: 'Q', 17: 'R', 18: 'S', 19: 'T', 20: 'U', 21: 'V', 22: 'W', 23: 'X', 24: 'Y', 25: 'Z'
}

while True:
    data_aux = []

    ret, frame = cap.read()

    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(frame_rgb)

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_drawing.draw_landmarks(
                frame,
                hand_landmarks,
                mp_hands.HAND_CONNECTIONS,
                mp_drawing_styles.get_default_hand_landmarks_style(),
                mp_drawing_styles.get_default_hand_connections_style()
            )

        for hand_landmarks in results.multi_hand_landmarks:
            for i in range(len(hand_landmarks.landmark)):
                    x = hand_landmarks.landmark[i].x
                    y = hand_landmarks.landmark[i].y

                    data_aux.append(x)
            
                    data_aux.append(y)
        
        prediction = model.predict([np.asarray(data_aux[:42])])
        if prediction[0] == "I Love You":
            predicted_character = "I Love You"
        else:
            predicted_character = labels_dict[int(prediction[0])]
        print(predicted_character)
        cv2.putText(frame, f"Prediction: {predicted_character}", 
                (50, 50),  
                cv2.FONT_HERSHEY_COMPLEX_SMALL,  
                1,  # Font scale
                (255, 255, 0),
                1,  # Thickness
                cv2.LINE_AA)  # Anti-aliasing

    cv2.imshow('frame', frame),
    cv2.waitKey(25)

cap.release()
cv2.destroyAllWindows()