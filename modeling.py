import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split

# Load data
data = pd.read_csv("data/train.csv")

# List labels
labels = ["toxic", "severe_toxic", "obscene", "threat", "insult", "identity_hate"]

# Data preprocessing
X = data["comment_text"]
y = data[labels]
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)
max_tokens = 200000
output_sequence_length = round(sum([len(comment.split()) for comment in data["comment_text"]]) / len(data))
vectorizer = tf.keras.layers.TextVectorization(max_tokens=max_tokens, 
                                               output_mode="int",
                                               output_sequence_length=output_sequence_length)

vectorizer.adapt(X_train)

# Modeling
tf.random.set_seed(42)

embedding_layer = tf.keras.layers.Embedding(input_dim=max_tokens,
                                            output_dim=128,
                                            embeddings_initializer="uniform",
                                            input_length=output_sequence_length)

inputs = tf.keras.layers.Input(shape=(1,), dtype="string")
layer = vectorizer(inputs)
layer = embedding_layer(layer)
layer = tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(64))(layer)
outputs = tf.keras.layers.Dense(6, activation="sigmoid")(layer)
model = tf.keras.Model(inputs, outputs)
model.compile(loss=tf.keras.losses.binary_crossentropy,
              optimizer=tf.keras.optimizers.Adam(),
              metrics=["accuracy"])

model.fit(X_train, y_train, epochs=5, validation_data=(X_test, y_test))

# Save model
model.save("model", save_format="tf")