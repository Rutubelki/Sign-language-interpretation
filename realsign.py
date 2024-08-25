import cv2
import numpy as np
import os
from matplotlib import pyplot as plt
import time
import mediapipe as mp
from scipy import stats

colors = [(245, 117, 16), (117, 245, 16), (16, 117, 245), (245, 66, 230),(233,34,56),(225,36,122),(57,78,122),(34,377,12),(23,455,12),(245,56,89)]

def prob_viz(res, actions, input_frame, colors):
    output_frame = input_frame.copy()
    for num, prob in enumerate(res):
        cv2.rectangle(output_frame, (0, 60 + num * 40), (int(prob * 100), 90 + num * 40), colors[num], -1)
        cv2.putText(output_frame, actions[num], (0, 85 + num * 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
    return output_frame

mp_holistic = mp.solutions.holistic # Holistic model
mp_drawing = mp.solutions.drawing_utils # Drawing utilities

def mediapipe_detection(image, model):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB) # COLOR-CONVERSION BGR-to-RGB
    image.flags.writeable = False                  # Convert image to not-writeable
    results = model.process(image)                 # Make prediction
    image.flags.writeable = True                   # Convert image to writeable 
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR) # COLOR-COVERSION RGB-to-BGR
    return image, results

def draw_landmarks(image, results):
    mp_drawing.draw_landmarks(image, results.face_landmarks, mp_holistic.FACEMESH_TESSELATION) # Draw face connections
    mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_holistic.POSE_CONNECTIONS) # Draw pose connections
    mp_drawing.draw_landmarks(image, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS) # Draw left hand connections
    mp_drawing.draw_landmarks(image, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS) # Draw right hand connections
    
def draw_styled_landmarks(image, results):
    # Draw face connections
    mp_drawing.draw_landmarks(image, results.face_landmarks, mp_holistic.FACEMESH_TESSELATION, 
                             mp_drawing.DrawingSpec(color=(80, 110, 10), thickness=1, circle_radius=1), 
                             mp_drawing.DrawingSpec(color=(80, 256, 121), thickness=1, circle_radius=1)) 
    # Draw pose connections
    mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_holistic.POSE_CONNECTIONS,
                             mp_drawing.DrawingSpec(color=(80, 22, 10), thickness=2, circle_radius=4), 
                             mp_drawing.DrawingSpec(color=(80, 44, 121), thickness=2, circle_radius=2)) 
    # Draw left hand connections
    mp_drawing.draw_landmarks(image, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS, 
                             mp_drawing.DrawingSpec(color=(121, 22, 76), thickness=2, circle_radius=4), 
                             mp_drawing.DrawingSpec(color=(121, 44, 250), thickness=2, circle_radius=2)) 
    # Draw right hand connections  
    mp_drawing.draw_landmarks(image, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS, 
                             mp_drawing.DrawingSpec(color=(245, 117, 66), thickness=2, circle_radius=4), 
                             mp_drawing.DrawingSpec(color=(245, 66, 230), thickness=2, circle_radius=2)) 
    
def extract_keypoints(results):
    pose = np.array([[res.x, res.y, res.z, res.visibility] for res in results.pose_landmarks.landmark]).flatten() if results.pose_landmarks else np.zeros(33*4)
    face = np.array([[res.x, res.y, res.z] for res in results.face_landmarks.landmark]).flatten() if results.face_landmarks else np.zeros(468*3)
    lh = np.array([[res.x, res.y, res.z] for res in results.left_hand_landmarks.landmark]).flatten() if results.left_hand_landmarks else np.zeros(21*3)
    rh = np.array([[res.x, res.y, res.z] for res in results.right_hand_landmarks.landmark]).flatten() if results.right_hand_landmarks else np.zeros(21*3)
    return np.concatenate([pose, face, lh, rh])

# Path for exported data, numpy arrays
DATA_PATH = os.path.join('MP_Data') 

# Actions that we try to detect
actions = np.array(['hello','eat','thank you','name','what','how'])

# Thirty videos worth of data
no_sequences = 30

# Videos are going to be 30 frames in length
sequence_length = 30

# Folder start
start_folder = 30

parent_folder = 'MP_Data'

from sklearn.model_selection import train_test_split
from keras import utils

label_map = {label: num for num, label in enumerate(actions)}

print(label_map)

sequences, labels = [], []
for action in actions:
    for sequence in np.array(os.listdir(os.path.join(DATA_PATH, action))).astype(int):
        window = []
        for frame_num in range(sequence_length):
            res = np.load(os.path.join(DATA_PATH, action, str(sequence), "{}.npy".format(frame_num)))
            window.append(res)
        sequences.append(window)
        labels.append(label_map[action])
        
X = np.array(sequences)
y = utils.to_categorical(labels).astype(int)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.05)

from keras import models
from keras import layers
from keras import callbacks

log_dir = os.path.join('Logs')
tb_callback = callbacks.TensorBoard(log_dir=log_dir)

model = models.Sequential()
model.add(layers.LSTM(64, return_sequences=True, activation='relu', input_shape=(30, 1662)))
model.add(layers.LSTM(128, return_sequences=True, activation='relu'))
model.add(layers.LSTM(64, return_sequences=False, activation='relu'))
model.add(layers.Dense(64, activation='relu'))
model.add(layers.Dense(32, activation='relu'))
model.add(layers.Dense(actions.shape[0], activation='softmax'))

model.compile(optimizer='Adam', loss='categorical_crossentropy', metrics=['categorical_accuracy'])

model.fit(X_train, y_train, epochs=700, callbacks=[tb_callback])

model.summary()

res = model.predict(X_test)
print(res[0])  # Debug print
print("Predicted action:", actions[np.argmax(res[3])])
print("True action:", actions[np.argmax(y_test[3])])

model.save('./model2.h5')
model.save_weights('.weights.h5')

del model

import tensorflow as tf
model = tf.keras.models.load_model('./model2.h5')
model.load_weights('.weights.h5')

from sklearn.metrics import multilabel_confusion_matrix, accuracy_score
yhat = model.predict(X_test)
ytrue = np.argmax(y_test, axis=1).tolist()
yhat = np.argmax(yhat, axis=1).tolist()

print("Confusion Matrix:", multilabel_confusion_matrix(ytrue, yhat))
print("Accuracy Score:", accuracy_score(ytrue, yhat))

# 1. New detection variables
sequence = []
sentence = []
predictions = []
threshold = 0.5

# 1. New detection variables
sequence = []
sentence = []
predictions = []
threshold = 0.5

cap = cv2.VideoCapture(0)
# Set mediapipe model 
with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
    while cap.isOpened():

        # Read feed
        ret, frame = cap.read()

        # Make detections
        image, results = mediapipe_detection(frame, holistic)
        print(results)
        
        # Draw landmarks
        draw_styled_landmarks(image, results)
        
        # 2. Prediction logic
        keypoints = extract_keypoints(results)
        sequence.append(keypoints)
        sequence = sequence[-30:]
        
        if len(sequence) == 30:
            res = model.predict(np.expand_dims(sequence, axis=0))[0]
            print(actions[np.argmax(res)])
            predictions.append(np.argmax(res))
            
            
        #3. Viz logic
            if np.unique(predictions[-10:])[0]==np.argmax(res): 
                if res[np.argmax(res)] > threshold: 
                    
                    if len(sentence) > 0: 
                        if actions[np.argmax(res)] != sentence[-1]:
                            sentence.append(actions[np.argmax(res)])
                    else:
                        sentence.append(actions[np.argmax(res)])

            if len(sentence) > 5: 
                sentence = sentence[-5:]

            # Viz probabilities
            image = prob_viz(res, actions, image, colors)
            
        cv2.rectangle(image, (0,0), (640, 40), (245, 117, 16), -1)
        cv2.putText(image, ' '.join(sentence), (3,30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
        
        # Show to screen
        cv2.imshow('OpenCV Feed', image)

        # Break gracefully
        if cv2.waitKey(10) & 0xFF == ord('q'):
            break
    cap.release()
    cv2.destroyAllWindows()


# import cv2
# import numpy as np
# import os
# from matplotlib import pyplot as plt
# import time
# import mediapipe as mp
# from scipy import stats
# from tkinter import *
# from PIL import Image, ImageTk

# colors = [(245, 117, 16), (117, 245, 16), (16, 117, 245), (245, 66, 230), (233, 34, 56), (225, 36, 122), (57, 78, 122), (34, 377, 12), (23, 455, 12), (245, 56, 89)]

# def prob_viz(res, actions, input_frame, colors):
#     output_frame = input_frame.copy()
#     for num, prob in enumerate(res):
#         cv2.rectangle(output_frame, (0, 60 + num * 40), (int(prob * 100), 90 + num * 40), colors[num], -1)
#         cv2.putText(output_frame, actions[num], (0, 85 + num * 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
#     return output_frame

# mp_holistic = mp.solutions.holistic  # Holistic model
# mp_drawing = mp.solutions.drawing_utils  # Drawing utilities

# def mediapipe_detection(image, model):
#     image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # COLOR-CONVERSION BGR-to-RGB
#     image.flags.writeable = False  # Convert image to not-writeable
#     results = model.process(image)  # Make prediction
#     image.flags.writeable = True  # Convert image to writeable
#     image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)  # COLOR-COVERSION RGB-to-BGR
#     return image, results

# def draw_landmarks(image, results):
#     mp_drawing.draw_landmarks(image, results.face_landmarks, mp_holistic.FACEMESH_TESSELATION)  # Draw face connections
#     mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_holistic.POSE_CONNECTIONS)  # Draw pose connections
#     mp_drawing.draw_landmarks(image, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS)  # Draw left hand connections
#     mp_drawing.draw_landmarks(image, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS)  # Draw right hand connections

# def draw_styled_landmarks(image, results):
#     # Draw face connections
#     mp_drawing.draw_landmarks(image, results.face_landmarks, mp_holistic.FACEMESH_TESSELATION,
#                               mp_drawing.DrawingSpec(color=(80, 110, 10), thickness=1, circle_radius=1),
#                               mp_drawing.DrawingSpec(color=(80, 256, 121), thickness=1, circle_radius=1))
#     # Draw pose connections
#     mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_holistic.POSE_CONNECTIONS,
#                               mp_drawing.DrawingSpec(color=(80, 22, 10), thickness=2, circle_radius=4),
#                               mp_drawing.DrawingSpec(color=(80, 44, 121), thickness=2, circle_radius=2))
#     # Draw left hand connections
#     mp_drawing.draw_landmarks(image, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS,
#                               mp_drawing.DrawingSpec(color=(121, 22, 76), thickness=2, circle_radius=4),
#                               mp_drawing.DrawingSpec(color=(121, 44, 250), thickness=2, circle_radius=2))
#     # Draw right hand connections
#     mp_drawing.draw_landmarks(image, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS,
#                               mp_drawing.DrawingSpec(color=(245, 117, 66), thickness=2, circle_radius=4),
#                               mp_drawing.DrawingSpec(color=(245, 66, 230), thickness=2, circle_radius=2))

# def extract_keypoints(results):
#     pose = np.array([[res.x, res.y, res.z, res.visibility] for res in results.pose_landmarks.landmark]).flatten() if results.pose_landmarks else np.zeros(33 * 4)
#     face = np.array([[res.x, res.y, res.z] for res in results.face_landmarks.landmark]).flatten() if results.face_landmarks else np.zeros(468 * 3)
#     lh = np.array([[res.x, res.y, res.z] for res in results.left_hand_landmarks.landmark]).flatten() if results.left_hand_landmarks else np.zeros(21 * 3)
#     rh = np.array([[res.x, res.y, res.z] for res in results.right_hand_landmarks.landmark]).flatten() if results.right_hand_landmarks else np.zeros(21 * 3)
#     return np.concatenate([pose, face, lh, rh])

# # Path for exported data, numpy arrays
# DATA_PATH = os.path.join('MP_Data')

# # Actions that we try to detect
# actions = np.array(['hello', 'eat', 'thank you', 'name', 'what', 'how'])

# # Thirty videos worth of data
# no_sequences = 30

# # Videos are going to be 30 frames in length
# sequence_length = 30

# # Folder start
# start_folder = 30

# parent_folder = 'MP_Data'

# from sklearn.model_selection import train_test_split
# from keras import utils

# label_map = {label: num for num, label in enumerate(actions)}

# print(label_map)

# sequences, labels = [], []
# for action in actions:
#     for sequence in np.array(os.listdir(os.path.join(DATA_PATH, action))).astype(int):
#         window = []
#         for frame_num in range(sequence_length):
#             res = np.load(os.path.join(DATA_PATH, action, str(sequence), "{}.npy".format(frame_num)))
#             window.append(res)
#         sequences.append(window)
#         labels.append(label_map[action])

# X = np.array(sequences)
# y = utils.to_categorical(labels).astype(int)

# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.05)

# from keras import models
# from keras import layers
# from keras import callbacks

# log_dir = os.path.join('Logs')
# tb_callback = callbacks.TensorBoard(log_dir=log_dir)

# model = models.Sequential()
# model.add(layers.LSTM(64, return_sequences=True, activation='relu', input_shape=(30, 1662)))
# model.add(layers.LSTM(128, return_sequences=True, activation='relu'))
# model.add(layers.LSTM(64, return_sequences=False, activation='relu'))
# model.add(layers.Dense(64, activation='relu'))
# model.add(layers.Dense(32, activation='relu'))
# model.add(layers.Dense(actions.shape[0], activation='softmax'))

# model.compile(optimizer='Adam', loss='categorical_crossentropy', metrics=['categorical_accuracy'])

# model.fit(X_train, y_train, epochs=100, callbacks=[tb_callback])

# model.summary()

# res = model.predict(X_test)
# print(res[0])  # Debug print
# print("Predicted action:", actions[np.argmax(res[3])])
# print("True action:", actions[np.argmax(y_test[3])])

# model.save('./model2.h5')
# model.save_weights('.weights.h5')

# del model

# import tensorflow as tf
# model = tf.keras.models.load_model('./model2.h5')
# model.load_weights('.weights.h5')

# from sklearn.metrics import multilabel_confusion_matrix, accuracy_score
# yhat = model.predict(X_test)
# ytrue = np.argmax(y_test, axis=1).tolist()
# yhat = np.argmax(yhat, axis=1).tolist()

# print("Confusion Matrix:", multilabel_confusion_matrix(ytrue, yhat))
# print("Accuracy Score:", accuracy_score(ytrue, yhat))

# # 1. New detection variables
# sequence = []
# sentence = []
# predictions = []
# threshold = 0.5

# # Tkinter GUI
# class SignLanguageApp:
#     def __init__(self, window, window_title):
#         self.window = window
#         self.window.title(window_title)

#         # Set the window size
#         self.window.geometry('800x600')  # Change size as needed

#         # Add a header label
#         self.header = Label(window, text="Sign Language Interpreter", font=('Arial', 24, 'bold'))
#         self.header.pack(side=TOP, fill=X)

#         self.video_source = 0
#         self.vid = cv2.VideoCapture(self.video_source)

#         # Create a canvas for video
#         self.canvas = Canvas(window, width=800, height=550)  # Adjust height to fit the header
#         self.canvas.pack()

#         self.delay = 15
#         self.sequence = []
#         self.sentence = []
#         self.predictions = []
#         self.colors = [(245, 117, 16), (117, 245, 16), (16, 117, 245)]

#         self.update()
#         self.window.mainloop()

#     def update(self):
#         ret, frame = self.vid.read()
#         if ret:
#             with mp_holistic.Holistic() as holistic:
#                 frame, results = mediapipe_detection(frame, holistic)
#                 draw_styled_landmarks(frame, results)

#                 keypoints = extract_keypoints(results)
#                 self.sequence.append(keypoints)
#                 self.sequence = self.sequence[-30:]

#                 if len(self.sequence) == 30:
#                     res = model.predict(np.expand_dims(self.sequence, axis=0))[0]
#                     self.predictions.append(np.argmax(res))

#                     if np.unique(self.predictions[-10:])[0] == np.argmax(res):
#                         if res[np.argmax(res)] > threshold:
#                             if len(self.sentence) > 0:
#                                 if actions[np.argmax(res)] != self.sentence[-1]:
#                                     self.sentence.append(actions[np.argmax(res)])
#                             else:
#                                 self.sentence.append(actions[np.argmax(res)])

#                     if len(self.sentence) > 5:
#                         self.sentence = self.sentence[-5:]

#                     # Viz probabilities
#                     frame = prob_viz(res, actions, frame, self.colors)

#             cv2.rectangle(frame, (0, 0), (800, 40), (245, 117, 16), -1)
#             cv2.putText(frame, ' '.join(self.sentence), (3, 30),
#                         cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)

#             self.photo = ImageTk.PhotoImage(image=Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)))
#             self.canvas.create_image(0, 0, image=self.photo, anchor=NW)

#         self.window.after(self.delay, self.update)

# SignLanguageApp(Tk(), "Sign Language Interpreter")