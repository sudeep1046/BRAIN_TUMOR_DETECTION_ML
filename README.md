
# Brain Tumor Detection Using VGG16

This repository contains code for detecting brain tumors from MRI images using a convolutional neural network based on the VGG16 architecture.

## Dataset

The dataset used in this project is the [Brain MRI Images for Brain Tumor Detection](https://www.kaggle.com/navoneel/brain-mri-images-for-brain-tumor-detection) from Kaggle. It contains MRI images categorized into two classes: `yes` for images with tumors and `no` for images without tumors.

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/your_username/brain-tumor-detection.git
   cd brain-tumor-detection
   ```

2. Download the dataset:
   ```bash
   mkdir -p ~/.kaggle/
   mv kaggle.json ~/.kaggle/
   chmod 600 ~/.kaggle/kaggle.json
   kaggle datasets download -d navoneel/brain-mri-images-for-brain-tumor-detection
   ```

3. Extract the dataset:
   ```python
   from zipfile import ZipFile

   file_name = "brain-mri-images-for-brain-tumor-detection.zip"
   with ZipFile(file_name,'r') as zip:
     zip.extractall()
     print('Done')
   ```

4. Install the required libraries:
   ```bash
   pip install -r requirements.txt
   ```

## Usage

1. Prepare the data:
   ```python
   import os
   import cv2
   from tqdm import tqdm
   import numpy as np
   from sklearn.model_selection import train_test_split
   from keras.utils import to_categorical
   from sklearn import preprocessing

   X = []
   y = []
   
   # Load images with tumors
   os.chdir('yes')
   for i in tqdm(os.listdir()):
       img = cv2.imread(i)
       img = cv2.resize(img, (224, 224))
       X.append(img)
       y.append('Y')

   # Load images without tumors
   os.chdir('../no')
   for i in tqdm(os.listdir()):
       img = cv2.imread(i)
       img = cv2.resize(img, (224, 224))
       X.append(img)
       y.append('N')
       
   X = np.array(X)
   y = np.array(y)
   
   le = preprocessing.LabelEncoder()
   y = le.fit_transform(y)
   y = to_categorical(y, num_classes=2)
   
   X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)
   ```

2. Train the model:
   ```python
   from keras.applications import vgg16
   from keras.models import Model
   from keras.layers import GlobalAveragePooling2D, Dense
   
   img_rows, img_cols = 224, 224
   
   vgg = vgg16.VGG16(weights='imagenet', include_top=False, input_shape=(img_rows, img_cols, 3))
   
   for layer in vgg.layers:
       layer.trainable = False

   def lw(bottom_model, num_classes):
       top_model = bottom_model.output
       top_model = GlobalAveragePooling2D()(top_model)
       top_model = Dense(1024, activation='relu')(top_model)
       top_model = Dense(1024, activation='relu')(top_model)
       top_model = Dense(512, activation='relu')(top_model)
       top_model = Dense(num_classes, activation='softmax')(top_model)
       return top_model

   num_classes = 2
   FC_Head = lw(vgg, num_classes)
   model = Model(inputs=vgg.input, outputs=FC_Head)

   model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
   history = model.fit(X_train, y_train, epochs=5, validation_data=(X_test, y_test), verbose=1)
   ```

3. Plot the results:
   ```python
   import matplotlib.pyplot as plt
   
   acc = history.history['accuracy']
   val_acc = history.history['val_accuracy']
   loss = history.history['loss']
   val_loss = history.history['val_loss']
   
   epochs = range(len(acc))
   
   plt.plot(epochs, acc, 'r', label='Training accuracy')
   plt.plot(epochs, val_acc, 'b', label='Validation accuracy')
   plt.title('Training and validation accuracy')
   plt.legend(loc=0)
   plt.figure()
   
   plt.show()
   ```

## Results

The model is trained for 5 epochs. The training and validation accuracy, as well as loss, are plotted to visualize the performance of the model.

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for more details.

## Acknowledgments

- [Kaggle](https://www.kaggle.com/) for providing the dataset.
- [TensorFlow](https://www.tensorflow.org/) and [Keras](https://keras.io/) for their excellent deep learning libraries.



