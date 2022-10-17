import numpy as np
import pandas as pd
import tensorflow as tf
import streamlit as st
import matplotlib.pyplot as plt

# Header
st.write("""
# Toxic Comment Classifier
This web app looks into classifying whether a comment is toxic or not.
Not just that, it will also try to classify its toxicity types.
So, go ahead and enter a comment below!
""")
st.write("*Disclaimer: this experimentation can still be further improve.*")
st.write("---")

# Load model in a singleton fashion
@st.experimental_singleton
def load_model():
  return tf.keras.models.load_model("model")

loaded_model = load_model()

# Comment input
comment_text = st.text_input("Enter a comment")

# Make predictions
pred_proba = loaded_model.predict([comment_text])

# Get prediction probabilities
labels_percentage = tf.squeeze(pred_proba)

# Get labels
labels = tf.cast(tf.squeeze(tf.round(pred_proba)), tf.int32)

# List toxicity types
types = ["toxic", "severe toxic", "obscene", "threat", "insult", "identity hate"]

# Setup prediction probabilities visualization
fig, ax = plt.subplots(figsize=(8, 5))
fig.suptitle("Prediction Probabilities")
ax.bar(types, labels_percentage.numpy().tolist(), color="c")
ax.set_ylim([0, 1])
ax.set_xlabel("toxicity types")
ax.set_ylabel("probabilities")

# Visualize the prediction probabilities for each type of toxicity
st.write("---")
st.write("""
### Predicition Probabilities Bar Chart Visualization
""")
st.pyplot(fig)

# Showcase classified toxicity types
st.write("---")
st.write("""
### Classified Toxicity Types
""")
for label, type in zip(labels, types):
  if label:
    st.write("The comment is considered to be **" + type + "**.")

st.write("---")