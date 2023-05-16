import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import time
import os
import cv2

from model import CVAE
from preprocess import load_dataset
from constants import *

dataset = load_dataset(
    save_name="dataset" if GRAY else "color_dataset"
)
print("Dataset shape:", dataset.shape)
dataset = (
    tf.data.Dataset.from_tensor_slices(dataset).shuffle(len(dataset)).batch(BATCH_SIZE)
)

prepath = "gray" if GRAY else "color"

optimizer = tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE)


def log_normal_pdf(sample, mean, logvar, raxis=1):
    log2pi = tf.math.log(2.0 * np.pi)
    return tf.reduce_sum(-0.5 * ((sample - mean) ** 2.0 * tf.exp(-logvar) + logvar + log2pi), axis=raxis)

def compute_loss(model, x):
    mean, logvar = model.encode(x)
    z = model.reparameterize(mean, logvar)
    x_logit = model.decode(z)
    cross_ent = tf.nn.sigmoid_cross_entropy_with_logits(logits=x_logit, labels=x)
    logpx_z = -tf.reduce_sum(cross_ent, axis=[1, 2, 3])
    logpz = log_normal_pdf(z, 0.0, 0.0)
    logqz_x = log_normal_pdf(z, mean, logvar)
    return -tf.reduce_mean(logpx_z + logpz - logqz_x)

@tf.function
def train_step(model, x, optimizer):
    with tf.GradientTape() as tape:
        loss = compute_loss(model, x)
    gradients = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))

model = CVAE(LATENT_DIM if GRAY else 3 * LATENT_DIM, color=not GRAY)

if LOAD_MODEL:
    dir_name = list(os.listdir(f"{prepath}/models"))[-1]
    saved_model = tf.keras.models.load_model(f"{prepath}/models/{dir_name}")
    model.set_weights(saved_model.get_weights())
    

def generate_and_save_images(model, epoch, test_sample):
    predictions = model(test_sample)
    fig = plt.figure(figsize=(4, 4))

    for i in range(predictions.shape[0]):
        plt.subplot(4, 4, i + 1)
        if GRAY:
            plt.imshow(predictions[i, :, :, 0], cmap='gray')
        else:
            img = predictions[i, :, :, :].numpy()
            b,g,r = cv2.split(img)
            rgb_img = cv2.merge([r,g,b])
            rgb_img = np.clip(rgb_img, 0, 1)
            plt.imshow(rgb_img)
        plt.axis('off')

    fig.suptitle(f"Epoch {epoch}")
    plt.savefig(f'{prepath}/epoch_imgs/image_at_epoch_{epoch:04d}.png')

def save_test_sample(test_sample):
    fig = plt.figure(figsize=(4, 4))

    for i in range(test_sample.shape[0]):
        plt.subplot(4, 4, i + 1)
        if GRAY:
            plt.imshow(test_sample[i, :, :, 0], cmap='gray')
        else:
            img = test_sample[i, :, :, :].numpy()
            b,g,r = cv2.split(img)
            rgb_img = cv2.merge([r,g,b])
            plt.imshow(rgb_img)
        plt.axis('off')

    fig.suptitle(f"Test Sample")
    plt.savefig(f'{prepath}/epoch_imgs/test_sample.png')

for test_batch in dataset.take(1):
    test_sample = test_batch[0:NUM_EXAMPLES_TO_GENERATE, :, :, :]

save_test_sample(test_sample)
generate_and_save_images(model, 0, test_sample)

for epoch in range(STARTING_EPOCH, EPOCHS + STARTING_EPOCH):
    start_time = time.time()
    for train_x in dataset:
        train_step(model, train_x, optimizer)
    end_time = time.time()

    loss = tf.keras.metrics.Mean()
    for test_x in dataset:
        loss(compute_loss(model, test_x))
    elbo = -loss.result()
    print('Epoch: {}, Dataset ELBO: {}, time elapse for current epoch: {}'
        .format(epoch, elbo, end_time - start_time))
    generate_and_save_images(model, epoch, test_sample)

model.save(f"{prepath}/models/model_{int(time.time())}")