import streamlit as st
import numpy as np
import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt
import gc

st.set_option('deprecation.showPyplotGlobalUse', False)

def main_section():
    st.title('Generative Adversarial Network project')
    st.markdown('This is an image generator based on the GANs trained on the 4 different datasets. '
                'Images can be generated in the **Image Generation** section.')


def load_models(dataset=None):
    if dataset == 'landscapes':
        trained_gan = keras.models.load_model('gan_landscapes')
        return trained_gan
    elif dataset == 'fingers':
        trained_gan = keras.models.load_model('gan_fingers')
        return trained_gan
    elif dataset == 'coins':
        trained_gan = keras.models.load_model('gan_2_coins')
        return trained_gan
    elif dataset == 'animals':
        trained_gan = keras.models.load_model('gan_animals')
        return trained_gan

def plot_images(images, n_cols=None):
    n_rows = len(images) / n_cols
    plt.figure(figsize=(15, 10))
    for index, image in enumerate(images):
        plt.subplot(n_rows, n_cols, index + 1)
        plt.imshow((image + 1) / 2)
        plt.axis('off')
    st.pyplot()


def generate_images():
    st.title('Image Generator')
    selected_dataset = st.sidebar.selectbox('Select Dataset', ['coins', 'fingers', 'animals', 'landscapes'])
    if st.sidebar.button('Generate image'):
        if selected_dataset == 'fingers':
            trained_gan = load_models(selected_dataset)
            noise = tf.random.normal(shape=[9, 100])
            gen_im = trained_gan(noise)
            plot_images(gen_im, n_cols=3)
            del trained_gan
        elif selected_dataset == 'animals':
            trained_gan = load_models(selected_dataset)
            noise = tf.random.normal(shape=[9, 100])
            gen_im = trained_gan(noise)
            plot_images(gen_im, n_cols=3)
            del trained_gan
        elif selected_dataset == 'coins':
            trained_gan = load_models(selected_dataset)
            noise = tf.random.normal(shape=[9, 300])
            gen_im = trained_gan(noise)
            plot_images(gen_im, n_cols=3)
            del trained_gan
        elif selected_dataset == 'landscapes':
            trained_gan = load_models(selected_dataset)
            noise = tf.random.normal(shape=[9, 300])
            gen_im = trained_gan(noise)
            plot_images(gen_im, n_cols=3)
            del trained_gan
activities = ['Main', 'Image Generation']
option = st.sidebar.selectbox('Select Option', activities)

if option == 'Main':
    main_section()

if option == 'Image Generation':
    generate_images()
    gc.collect()