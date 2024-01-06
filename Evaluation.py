


import cv2
from keras.models import model_from_json
import numpy as np


# In[48]:


# Load the Keras model architecture from JSON
json_file = open("/Users/faheem/Downloads/Furhat-chatbot/facialemotionmodel.json", "r")
model_json = json_file.read()
json_file.close()

# Load the Keras model weights
model = model_from_json(model_json)
model.load_weights("/Users/faheem/Downloads/Furhat-chatbot/facialemotionmodel.h5")


# In[49]:


# Initialize Haar Cascade for face detection
haar_file = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
face_cascade = cv2.CascadeClassifier(haar_file)

# Function to extract features from an image
def extract_features(image):
    feature = np.array(image)
    feature = feature.reshape(1, 48, 48, 1)
    return feature / 255.0


# In[50]:


# Open the webcam
webcam = cv2.VideoCapture(0)

# Define emotion labels
labels = {0: 'angry', 1: 'disgust', 2: 'fear', 3: 'happy', 4: 'neutral', 5: 'sad', 6: 'surprise'}


# In[51]:


# Main loop for video capture and emotion prediction
while True:
    ret, frame = webcam.read()
    if not ret:
        print("Error reading frame from the webcam.")
        break

    # Convert the frame to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect faces in the frame
    faces = face_cascade.detectMultiScale(frame, 1.3, 5)

    try:
        for (x, y, w, h) in faces:
            # Extract the face region
            face_image = gray[y:y + h, x:x + w]

            # Draw a rectangle around the face
            cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)

            # Resize the face image to 48x48 pixels
            face_image = cv2.resize(face_image, (48, 48))

            # Extract features and preprocess for emotion prediction
            img = extract_features(face_image)

            # Use the pre-trained model to predict the emotion
            pred = model.predict(img)

            # Get the predicted emotion label
            prediction_label = labels[pred.argmax()]

            # Display the predicted emotion label on the frame
            cv2.putText(frame, prediction_label, (x - 10, y - 10), cv2.FONT_HERSHEY_COMPLEX_SMALL, 2, (0, 0, 255))
        
        # Display the output frame
        cv2.imshow("Facial Emotion Recognition", frame)

        # Break the loop if 'Esc' key is pressed
        key = cv2.waitKey(1) & 0xFF
        if key == 27:
            break

    except cv2.error:
        pass
    
# Release the webcam and close all windows
webcam.release()
cv2.destroyAllWindows()


# In[52]:


# Release the webcam and close all windows
webcam.release()
cv2.destroyAllWindows()


# In[ ]:




