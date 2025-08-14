import os
import numpy as np
import keras
from keras.models import Sequential
from keras.layers import Dense, Flatten, BatchNormalization, Conv2D, MaxPooling2D
from keras.optimizers import Adam
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import itertools

# Check and set dataset paths
if "google.colab" in str(get_ipython()):
    from google.colab import drive
    drive.mount('/content/drive')
    base_path = '/content/drive/MyDrive/path_to_dataset'  # Update this path
else:
    base_path = os.getcwd()

train_path = os.path.join(base_path, 'train')
valid_path = os.path.join(base_path, 'valid')
test_path = os.path.join(base_path, 'test')

# Define classes

classes = ['no_dr', 'mild_dr', 'moderate_dr', 'severe_dr', 'proliferative_dr']

# Function to check dataset structure
def check_dataset_structure(base_path, class_names):
    if not os.path.exists(base_path):
        raise ValueError(f"Dataset path '{base_path}' does not exist!")
    for class_name in class_names:
        class_path = os.path.join(base_path, class_name)
        if not os.path.exists(class_path) or len(os.listdir(class_path)) == 0:
            raise ValueError(f"No images found in '{class_path}'. Check directory structure.")

# Validate dataset structure
check_dataset_structure(train_path, classes)
check_dataset_structure(valid_path, classes)
check_dataset_structure(test_path, classes)

# Load and preprocess dataset
train_batches = ImageDataGenerator(rescale=1./255).flow_from_directory(
    train_path, target_size=(224,224), classes=classes, batch_size=10, class_mode='categorical'
)
valid_batches = ImageDataGenerator(rescale=1./255).flow_from_directory(
    valid_path, target_size=(224,224), classes=classes, batch_size=10, class_mode='categorical'
)
test_batches = ImageDataGenerator(rescale=1./255).flow_from_directory(
    test_path, target_size=(224,224), classes=classes, batch_size=10, class_mode='categorical'
)

# Define CNN Model
model = Sequential([
    Conv2D(32, (3,3), activation='relu', input_shape=(224,224,3)),
    MaxPooling2D(pool_size=(2,2)),
    BatchNormalization(),

    Conv2D(64, (3,3), activation='relu'),
    MaxPooling2D(pool_size=(2,2)),
    BatchNormalization(),

    Conv2D(128, (3,3), activation='relu'),
    MaxPooling2D(pool_size=(2,2)),
    BatchNormalization(),

    Flatten(),
    Dense(512, activation='relu'),
    Dense(5, activation='softmax')  # 5 classes
])

# Compile the model
model.compile(optimizer=Adam(learning_rate=0.0001),
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# Train the model
model.fit(train_batches,
          validation_data=valid_batches,
          epochs=10,
          verbose=1)

# Evaluate model on test data
test_loss, test_acc = model.evaluate(test_batches, verbose=1)
print(f'Test Accuracy: {test_acc * 100:.2f}%')