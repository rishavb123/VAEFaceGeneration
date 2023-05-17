import tensorflow as tf
import time
import os
import numpy as np

from model import CVAE

import cv2

gray = False
latent_dim = 100
load_eps = True
row, col = 4, 8
img_size = 256
gray_divider = 15
color_divider = 5
# Set row, col to 1, 1 to enable saving


full_num = row * col
latent_dim = latent_dim if gray else 3 * latent_dim

model = CVAE(latent_dim, color=not gray)

prepath = "gray" if gray else "color"

dir_name = list(os.listdir(f"{prepath}/models"))[-1]
saved_model = tf.keras.models.load_model(f"{prepath}/models/{dir_name}")
model.set_weights(saved_model.get_weights())

eps_names = list(os.listdir(f"{prepath}/eps"))
if len(eps_names) == 0 or not load_eps:
    eps = tf.random.normal(shape=(full_num, latent_dim))
else:
    eps = tf.convert_to_tensor(
        np.repeat(np.load(f"{prepath}/eps/{eps_names[-1]}"), full_num, axis=0)
    )
divider = gray_divider if gray else color_divider

while True:
    d_eps = tf.random.normal(shape=(full_num, latent_dim)) / divider
    eps += d_eps

    results = model.sample(eps).numpy()

    imgs = []

    for r in range(row):
        imgs.append([])
        for c in range(col):
            i = r * col + c
            imgs[-1].append(results[i])

    imgs = [np.concatenate(img_row, axis=1) for img_row in imgs]

    img = np.concatenate(imgs, axis=0)

    img = cv2.resize(img, (img_size * col, img_size * row))

    cv2.imshow("VAE", img)

    if cv2.waitKey(1) == ord("q"):
        break

    if cv2.waitKey(1) == ord("s") and row == 1 and col == 1:
        np.save(f"{prepath}/eps/eps_{int(time.time())}.npy", eps.numpy())
