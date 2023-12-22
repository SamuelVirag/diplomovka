from keras import layers, models
from keras.utils import np_utils
from sklearn.model_selection import train_test_split
import os
import numpy as np
from sklearn.preprocessing import OneHotEncoder
import tensorflow as tf


# Load your spectrogram data
def load_data(data_path):
    data = []
    labels = []

    for speaker_folder in os.listdir(data_path):
        speaker_path = os.path.join(data_path, speaker_folder)

        for file in os.listdir(speaker_path):
            file_path = os.path.join(speaker_path, file)

            # Load spectrogram from .npy file
            spectrogram = np.load(file_path)
            #print(spectrogram.shape)
            spectrogram = (spectrogram - np.min(spectrogram)) / (np.max(spectrogram) - np.min(spectrogram))

            # Assuming the folder name is the speaker label
            label = int(speaker_folder)
            labels.append(label)
            data.append(spectrogram)

    print(len(labels))
    print(len(data))
    return np.array(data), np.array(labels)


# Specify the paths to your train and test data
train_data_path = os.path.join('data', 'train-data-spectrogram', 'tf', 'mel')
test_data_path = os.path.join('data', 'test-data-spectrogram', 'tf', 'mel')
val_data_path = os.path.join('data', 'val-data-spectrogram', 'tf', 'mel')

# Load train and test data
X_train, y_train = load_data(train_data_path)
X_test, y_test = load_data(test_data_path)
X_val, y_val = load_data(val_data_path)

# Normalize data
X_train_normalized = (X_train - np.min(X_train)) / (np.max(X_train) - np.min(X_train))
X_test_normalized = (X_test - np.min(X_test)) / (np.max(X_test) - np.min(X_test))
X_val_normalized = (X_val - np.min(X_val)) / (np.max(X_val) - np.min(X_val))

# Reshape data to include the channel dimension
X_train_normalized = X_train_normalized.reshape(X_train_normalized.shape + (1,))
X_test_normalized = X_test_normalized.reshape(X_test_normalized.shape + (1,))
X_val_normalized = X_val_normalized.reshape(X_val_normalized.shape + (1,))

# One-hot encode labels
encoder = OneHotEncoder(sparse=False)
y_train_encoded = encoder.fit_transform(y_train.reshape(-1, 1))
y_test_encoded = encoder.transform(y_test.reshape(-1, 1))
y_val_encoded = encoder.transform(y_val.reshape(-1, 1))

# # Split data into train and validation sets
# X_train_final, X_val_final, y_train_final, y_val_final = train_test_split(X_train_normalized, y_train_encoded, test_size=0.2, random_state=42)

# Assuming your spectrogram shape is (67, 128)
spectrogram_height, spectrogram_width = 67, 128

# Define the CNN model
model = models.Sequential()
model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(spectrogram_height, spectrogram_width, 1)))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(64, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(64, (3, 3), activation='relu'))
model.add(layers.Flatten())
model.add(layers.Dense(64, activation='relu'))
model.add(layers.Dense(40, activation='softmax'))  # Number of classes is the number of unique speakers

# Compile the model
model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# Display the model summary
model.summary()

# Train the model
history = model.fit(X_train_normalized, y_train_encoded, epochs=10, validation_data=(X_val_normalized, y_val_encoded))

# Evaluate the model on the test set
test_loss, test_acc = model.evaluate(X_test_normalized, y_test_encoded)
print(f'Test accuracy: {test_acc}')
