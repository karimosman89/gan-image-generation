import numpy as np
from tensorflow.keras.models import load_model

def load_gan_generator(model_path):
    """Load the pre-trained GAN generator."""
    return load_model(model_path)

def generate_noise(dim, num_samples):
    """Generate random noise as input for the GAN."""
    return np.random.normal(0, 1, (num_samples, dim))

def generate_images(generator, noise):
    """Generate images using the GAN generator."""
    return generator.predict(noise)

