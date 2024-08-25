# Sign-language-interpretation

Overview
This project is focused on developing a Sign Language Interpretation system using deep learning(using LSTM) and computer vision techniques. The system detects and interprets sign language gestures into text using Google’s MediaPipe for landmark detection and custom datasets.

Project Structure
creating_dataset.py: Script for creating and recording the custom dataset.
realsign.py: Script for real-time gesture recognition and interpretation.

Before running the scripts, make sure you have the following installed:

Python 3.x
OpenCV
MediaPipe
Numpy
Pandas
You can install the required Python packages using the following command:

How to Use
Step 1: Create the Dataset
To create your own dataset for training the model, run the creating_dataset.py script. This script will guide you through recording various sign language gestures using your webcam. The recorded data will be stored in the /dataset directory.

python creating_dataset.py
Step 2: Run the Real-Time Gesture Recognition
After creating the dataset, you can start the real-time gesture recognition by running the realsign.py script. This script uses the recorded dataset and MediaPipe for landmark detection to interpret sign language gestures in real-time.

python realsign.py
Step 3: Customizing and Training the Model
If you want to customize the model or improve its accuracy, you can modify the training scripts and the model architecture. The model training process can be tailored to your specific dataset.

Project Flow
Data Collection: Use creating_dataset.py to capture gesture data.
Model Training: (Optional) Customize and train your model using the recorded data.
Real-Time Recognition: Use realsign.py to recognize and interpret gestures in real-time.
Acknowledgements
This project was built using:

Google's MediaPipe for landmark detection.
OpenCV for computer vision tasks.
