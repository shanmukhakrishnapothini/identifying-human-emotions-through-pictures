# Import Libraries
import os
import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from keras.preprocessing.image import ImageDataGenerator
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix

# Step 1: Load and Preprocess Data
def load_data(data_path):
    images = []
    labels = []
    for emotion in os.listdir(data_path):
        emotion_path = os.path.join(data_path, emotion)
        for img in os.listdir(emotion_path):
            img_path = os.path.join(emotion_path, img)
            image = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
            image = cv2.resize(image, (48, 48))  # Resize to 48x48 pixels
            images.append(image)
            labels.append(emotion)
    images = np.array(images).reshape(-1, 48, 48, 1) / 255.0  # Normalize
    return images, np.array(labels)

# Load dataset
data_path = 'C:\Users\Hp\emotion_recognition\emotion_dataset\train'  # Update with your dataset path
X, y = load_data(data_path)

# Step 2: Build the CNN Model
def build_model():
    model = Sequential()
    model.add(Conv2D(32, (3, 3), activation='relu', input_shape=(48, 48, 1)))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Conv2D(64, (3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Conv2D(128, (3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Flatten())
    model.add(Dense(128, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(6, activation='softmax'))  # 6 emotions
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    return model

model = build_model()

# Step 3: Split Data for Training and Validation
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 4: Data Augmentation
datagen = ImageDataGenerator(rotation_range=20, width_shift_range=0.1,
                             height_shift_range=0.1, shear_range=0.1,
                             zoom_range=0.1, horizontal_flip=True)

# Step 5: Train the Model
model.fit(datagen.flow(X_train, y_train, batch_size=32), 
          validation_data=(X_val, y_val), epochs=50)

# Step 6: Evaluate the Model
y_pred = np.argmax(model.predict(X_val), axis=-1)
print(classification_report(y_val, y_pred))

# Confusion Matrix
conf_matrix = confusion_matrix(y_val, y_pred)
sns.heatmap(conf_matrix, annot=True, fmt='d')
plt.ylabel('Actual')
plt.xlabel('Predicted')
plt.title('Confusion Matrix')
plt.show()

# Step 7: Make Predictions
def predict_emotion(image_path):
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    image = cv2.resize(image, (48, 48)) / 255.0
    image = np.expand_dims(image, axis=0).reshape(-1, 48, 48, 1)
    prediction = model.predict(image)
    emotion = np.argmax(prediction)
    return emotion  # Map this to actual emotion labels

# Example usage
new_image_path = 'C:\Users\Hp\emotion_recognition\emotion_dataset\test'  # Update with your image path
predicted_emotion = predict_emotion(new_image_path)
print("Predicted Emotion:", predicted_emotion)
