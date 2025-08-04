✨ Task 4: Hand Gesture Recognition using CNN
SkillCraft Machine Learning Internship

This project focuses on developing a CNN model to recognize various hand gestures using the LeapGestRecog dataset.
The model is trained on 10 classes like palm, fist, index, ok, down, etc., and can predict gestures from unseen images.

🛠️ Model Summary:
✅ Convolutional Neural Network (CNN)

✅ Trained on grayscale gesture images (64x64)

✅ 10 gesture classes

✅ Accuracy: ~XX% (replace with yours)

📊 Sample Predictions
Below are predictions made by the trained model on test samples:


🧠 Technologies Used
Python, NumPy, Matplotlib

TensorFlow / Keras

Google Colab

📂 Dataset
LeapGestRecog dataset (from Kaggle):
https://www.kaggle.com/datasets/kmader/leapgestrecog

🤖 Model Demo
To see predictions on test images:
model = load_model('gesture_model.keras')

# Preprocess image & predict as shown in notebook
📁 Files

**File**	                                        **Description**

gesture_model.keras	                         Trained model
gesture_predictions.png	                     Sample predictions
hand_gesture_recognition.ipynb	             Training and testing notebook
README.md                                  	 Project overview

