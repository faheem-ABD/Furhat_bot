#!/usr/bin/env python
# coding: utf-8

# In[71]:


from keras.utils import to_categorical
from keras_preprocessing.image import load_img
from keras.models import Sequential
from keras.layers import Dense, Conv2D, Dropout, Flatten, MaxPooling2D
import os
import pandas as pd
import numpy as np
from tqdm.notebook import tqdm

from sklearn.preprocessing import LabelEncoder


# In[72]:


TRAIN_DIR = '/Users/faheem/Downloads/Furhat-chatbot/images/train'
TEST_DIR = '/Users/faheem/Downloads/Furhat-chatbot/images/test'


# In[73]:


def createdataframe(dir):
    image_paths = []
    labels = []

    for label in os.listdir(dir):
        label_path = os.path.join(dir, label)

        # To Check if it's a directory
        if os.path.isdir(label_path):
            for imagename in os.listdir(label_path):
                image_paths.append(os.path.join(label_path, imagename))
                labels.append(label)
            print(label, "completed")

    return image_paths, labels



# In[74]:


train = pd.DataFrame()
train['image'], train['label'] = createdataframe(TRAIN_DIR)
print(train)


# In[75]:


test = pd.DataFrame()
test['image'], test['label'] = createdataframe(TEST_DIR)

print(test)
print(test['image'])



# In[76]:


def extract_features(images):
    features = []
    for image in tqdm(images):
        img = load_img(image,grayscale = True)  #grayscale with 48 x 48 is best for image analysis training and 1 for 2d images
        img = np.array(img)
        features.append(img)
    features = np.array(features) 
    features = features.reshape(len(features),48,48,1)
    return features


# In[77]:


train_features = extract_features(train['image'])


# In[78]:


test_features = extract_features(test['image'])


# In[81]:


x_train = train_features/255.0
x_test = test_features/255.0


# In[82]:


le = LabelEncoder()
le.fit(train['label'])


# In[83]:


y_train = le.transform(train['label'])
y_test = le.transform(test['label'])


# In[84]:


y_train = to_categorical(y_train, num_classes = 7)
y_test = to_categorical(y_test, num_classes = 7)


# In[86]:


# Explanation: Add a 2D convolutional layer with 128 filters, each of size (3, 3), using ReLU activation.
#              Input shape is set to (48, 48, 1), suitable for grayscale images.
# Explanation: Add a 2D max pooling layer with pool size (2, 2) to reduce spatial dimensions.
# Explanation: Add a dropout layer with a dropout rate of 0.4, which helps prevent overfitting during training.
# Explanation: Add another convolutional layer with 256 filters, each of size (3, 3), using ReLU activation.
# Explanation: Add another max pooling layer to further reduce spatial dimensions.
# Explanation: Add another dropout layer with a dropout rate of 0.4.
# Explanation: Add another convolutional layer with 512 filters, each of size (3, 3), using ReLU activation.

 # Explanation: Add another max pooling layer.
    # Explanation: Add another dropout layer with a dropout rate of 0.4.
    # Explanation: Add another convolutional layer with 512 filters, each of size (3, 3), using ReLU activation.
    # Explanation: Add another max pooling layer.
# Explanation: Add another dropout layer with a dropout rate of 0.4.
# Flatten layer to transition from convolutional to fully connected layers
# Explanation: Flatten the output of the last convolutional layer into a one-dimensional vector.
# Fully connected layers
# Explanation: Add a fully connected layer with 512 neurons using ReLU activation.
# Explanation: Add a dropout layer with a dropout rate of 0.4.
# Explanation: Add another fully connected layer with 256 neurons using ReLU activation.
# Explanation: Add another dropout layer with a dropout rate of 0.3.
# Output layer with 7 units (assuming a 7-class classification problem) and softmax activation
 #Explanation: Add the output layer with 7 neurons (assuming 7 classes) using softmax activation for multi-class classification.


# In[87]:


model = Sequential()

#CNN

# Convolutional layers
model.add(Conv2D(128, kernel_size=(3, 3), activation='relu', input_shape=(48, 48, 1)))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.4))

model.add(Conv2D(256, kernel_size=(3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.4))

model.add(Conv2D(512, kernel_size=(3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.4))

model.add(Conv2D(512, kernel_size=(3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.4))

model.add(Flatten())
model.add(Dense(512, activation='relu'))
model.add(Dropout(0.4))
model.add(Dense(256, activation='relu'))
model.add(Dropout(0.3))

#output layer
model.add(Dense(7, activation='softmax'))


# In[88]:


model.compile(optimizer = 'adam', loss = 'categorical_crossentropy', metrics = 'accuracy' )


# In[89]:


model.fit(x= x_train,y = y_train, batch_size = 128, epochs = 100, validation_data = (x_test,y_test)) 


# In[92]:


model_json = model.to_json()
with open("emotiondetector.json",'w') as json_file:
    json_file.write(model_json)
model.save("emotiondetector.h5")


# In[93]:


from keras.models import model_from_json

# Save model as .h5 file
model.save('/Users/faheem/Downloads/emotiondetector.h5')

# Save model architecture as .json file
model_json = model.to_json()
with open('/Users/faheem/Downloads/emotiondetector.json', 'w') as json_file:
    json_file.write(model_json)


# In[94]:


# Evaluate the model on the training set
train_metrics = model.evaluate(x_train, y_train)
print("Training Accuracy:", train_metrics[1])


# In[95]:


# Evaluate the model on the test set
test_metrics = model.evaluate(x_test, y_test)
print("Test Accuracy:", test_metrics[1])


# In[99]:


from keras.models import load_model
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from keras_preprocessing.image import load_img
from keras.utils import to_categorical
from keras.models import Sequential
from keras.layers import Dense, Conv2D, Dropout, Flatten, MaxPooling2D
import os
from sklearn.preprocessing import LabelEncoder


# In[103]:


# Set the path to the model file and the Downloads directory
model_file_path = '/Users/faheem/Downloads/emotiondetector.h5'
downloads_path = '/Users/faheem/Downloads/'


# In[104]:


# Load the model
model = load_model(model_file_path)

# Labels for emotions
label = ['angry', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprise']



# In[105]:


def ef(image):
    img = load_img(image, grayscale=True)
    feature = np.array(img)
    feature = feature.reshape(1, 48, 48, 1)
    return feature / 255.0


# In[31]:


# Labels for emotions
label = ['angry', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprise']

def ef(image):
    img = load_img(image, grayscale=True)
    feature = np.array(img)
    feature = feature.reshape(1, 48, 48, 1)
    return feature / 255.0


# In[106]:


# Image paths in the 'images' folder
image_paths = [
    'images/train/sad/42.jpg',
    'images/train/fear/2.jpg',
    'images/train/disgust/299.jpg',
    'images/train/happy/7.jpg',
    'images/train/surprise/15.jpg'
]


# In[108]:


# Display predictions for each image
for image_path in image_paths:
    full_image_path = os.path.join(downloads_path, image_path)
    emotion_label = image_path.split('/')[2]  # Extracting emotion label from path
    print(f"Original image is of {emotion_label}")

    img = ef(full_image_path)
    pred = model.predict(img)
    pred_label = label[pred.argmax()]

    print(f"Model prediction is {pred_label}")
    plt.imshow(img.reshape(48, 48), cmap='gray')
    plt.show()


# In[ ]:




