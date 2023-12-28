import tensorflow as tf
from tensorflow.keras.layers import Dense, Input
from tensorflow.keras.models import Model

# Simple Attention Layer
class AttentionLayer(tf.keras.layers.Layer):
    def __init__(self, **kwargs):
        super(AttentionLayer, self).__init__(**kwargs)

    def build(self, input_shape):
        self.W = self.add_weight(name='attention_weight',
                                 shape=(input_shape[-1], 1),
                                 initializer='random_normal',
                                 trainable=True)
        super(AttentionLayer, self).build(input_shape)

    def call(self, x):
        e = tf.keras.backend.tanh(tf.keras.backend.dot(x, self.W))
        a = tf.keras.backend.softmax(e, axis=1)
        output = x * a
        return tf.keras.backend.sum(output, axis=1)

# Sample data
import numpy as np
data = np.random.rand(100, 90)  # 100 samples, 90 features
labels = np.random.rand(100, 1)  # 100 labels

# Input Layer
input_layer = Input(shape=(90,))

# Attention Layer
attention_output = AttentionLayer()(input_layer)

# Dense Layers (using attention output directly)
dense_layer = Dense(64, activation='relu')(attention_output)
output_layer = Dense(1, activation='linear')(dense_layer)  # Change activation for classification

# Create Model
model = Model(inputs=input_layer, outputs=output_layer)

# Compile Model
model.compile(optimizer='adam', loss='mean_squared_error')  # Change loss for classification

# Model Summary
model.summary()

# Train Model
model.fit(data, labels, epochs=10, batch_size=32)
