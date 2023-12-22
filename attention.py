import os
from tensorflow import keras

os.environ['TF_GPU_ALLOCATOR'] = 'cuda_malloc_async'
from keras import layers, models
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

# Define the MultiHeadSelfAttention layer
class MultiHeadSelfAttentionWithPooling(layers.Layer):
    def __init__(self, embed_dim, num_heads=1):
        super(MultiHeadSelfAttentionWithPooling, self).__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads

        assert (
            self.head_dim * num_heads == embed_dim
        ), "Embedding dimension needs to be divisible by the number of heads"

        self.query_dense = tf.keras.layers.Dense(embed_dim)
        self.key_dense = tf.keras.layers.Dense(embed_dim)
        self.value_dense = tf.keras.layers.Dense(embed_dim)
        self.combine_heads = tf.keras.layers.Dense(embed_dim)

        # Pooling layer
        self.pooling = tf.keras.layers.MaxPooling1D(pool_size=2, strides=2, padding="same")

    def attention(self, query, key, value):
        score = tf.matmul(query, key, transpose_b=True)
        dim_key = tf.cast(tf.shape(key)[-1], tf.float32)
        scaled_score = score / tf.math.sqrt(dim_key)

        weights = tf.nn.softmax(scaled_score, axis=-1)

        output = tf.matmul(weights, value)
        return output, weights

    def separate_heads(self, x, batch_size):
        x = tf.reshape(x, (batch_size, -1, self.num_heads, self.head_dim))
        x = tf.transpose(x, perm=[0, 2, 1, 3])

        # Reshape to remove unnecessary dimension
        x = tf.reshape(x, (batch_size * self.num_heads, -1, self.head_dim))

        # Apply pooling
        x = self.pooling(x)

        return x

    def call(self, inputs):
        # Assuming inputs is of shape (batch_size, seq_len, embed_dim)

        batch_size = tf.shape(inputs)[0]

        # Linear projections
        query = self.query_dense(inputs)
        key = self.key_dense(inputs)
        value = self.value_dense(inputs)

        # Separate heads
        query = self.separate_heads(query, batch_size)
        key = self.separate_heads(key, batch_size)
        value = self.separate_heads(value, batch_size)

        # Apply attention
        attention, weights = self.attention(query, key, value)

        # Pooling
        attention_pooled = self.pooling(attention)

        # Reshape and combine heads
        attention_pooled = tf.reshape(attention_pooled, (batch_size, -1, self.head_dim * self.num_heads))
        output = self.combine_heads(attention_pooled)

        return output

# Define the TransformerBlock
class TransformerBlock(layers.Layer):
    def __init__(self, embed_dim, num_heads, ff_dim, rate=0.1):
        super(TransformerBlock, self).__init__()
        self.att = MultiHeadSelfAttentionWithPooling(embed_dim, num_heads)
        self.ffn = keras.Sequential(
            [layers.Dense(ff_dim, activation="relu"), layers.Dense(embed_dim),]
        )
        self.layernorm1 = layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = layers.LayerNormalization(epsilon=1e-6)
        self.dropout1 = layers.Dropout(rate)
        self.dropout2 = layers.Dropout(rate)

    def call(self, inputs):
        attention_output = self.att(inputs)  # No need for the mask argument
        x = self.dropout1(attention_output)
        x = tf.tile(x, [1, 4, 1])
        print(inputs.shape)
        print(x.shape)
        out1 = self.layernorm1(inputs + x)

        ffn_output = self.ffn(out1)
        x = self.dropout2(ffn_output)
        return self.layernorm2(out1 + x)

# Define the AttentionSpeakerVerificationModel
class AttentionSpeakerVerificationModel(models.Model):
    def __init__(self, embed_dim, num_heads, ff_dim, num_classes, rate=0.1):
        super(AttentionSpeakerVerificationModel, self).__init__()
        self.attention = MultiHeadSelfAttentionWithPooling(embed_dim, num_heads)
        self.transformer_block = TransformerBlock(embed_dim, num_heads, ff_dim, rate)
        self.global_avg_pooling = layers.GlobalAveragePooling1D()
        self.dense = layers.Dense(num_classes, activation="softmax")

    def call(self, inputs):
        x = self.attention(inputs)
        x = self.transformer_block(x)  # Remove the mask argument here
        x = self.global_avg_pooling(x)
        return self.dense(x)

# Now, create an instance of the AttentionSpeakerVerificationModel
attention_model = AttentionSpeakerVerificationModel(
    embed_dim=128, num_heads=1, ff_dim=32, num_classes=40
)

# Explicitly call the model on a batch of data to trigger the build process
dummy_input = tf.constant(X_train_normalized[:1], dtype=tf.float32)
_ = attention_model(dummy_input)

# Compile the model
attention_model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Display the model summary
attention_model.summary()

# Train the model
history = attention_model.fit(X_train_normalized, y_train_encoded, epochs=10, validation_data=(X_val_normalized, y_val_encoded))

# Evaluate the model on the test set
test_loss, test_acc = attention_model.evaluate(X_test_normalized, y_test_encoded)

print(f'Test accuracy: {test_acc}')