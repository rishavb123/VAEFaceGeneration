import tensorflow as tf
import time
import os
import numpy as np

from model import CVAE

import cv2

gray = False
latent_dim = 100
load_eps = False


latent_dim = latent_dim if gray else 3 * latent_dim

model = CVAE(latent_dim, color=not gray)

prepath = "gray" if gray else "color"

dir_name = list(os.listdir(f"{prepath}/models"))[-1]
saved_model = tf.keras.models.load_model(f"{prepath}/models/{dir_name}")
model.set_weights(saved_model.get_weights())

eps_names = list(os.listdir(f"{prepath}/eps"))
if len(eps_names) == 0 or not load_eps:
    eps = tf.random.normal(shape=(1, latent_dim))
else:
    eps = tf.convert_to_tensor(np.load(f"{prepath}/eps/{eps_names[-1]}"))
divider = 15 if gray else 10

while True:
    d_eps = tf.random.normal(shape=(1, latent_dim)) / divider
    eps += d_eps

    img = model.sample(eps)[0].numpy()

    img = cv2.resize(img, (512, 512))

    cv2.imshow("VAE", img)

    if cv2.waitKey(1) == ord("q"):
        break

    if cv2.waitKey(1) == ord("s"):
        np.save(f"{prepath}/eps/eps_{int(time.time())}.npy", eps.numpy())
