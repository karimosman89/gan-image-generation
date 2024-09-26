import streamlit as st
import numpy as np
from tensorflow.keras.models import load_model
from PIL import Image

# Load the GAN generator model
generator = load_model('generator_model.h5')

st.title("Generate Images using GAN")

# Slider for the number of images to generate
num_images = st.slider("Number of images to generate:", 1, 10)

if st.button("Generate"):
    noise = np.random.normal(0, 1, (num_images, 100))
    generated_images = generator.predict(noise)

    st.write("Generated Images:")
    for img in generated_images:
        img = (img * 127.5 + 127.5).astype(np.uint8)  # Rescale image pixels
        st.image(Image.fromarray(img), width=128)

