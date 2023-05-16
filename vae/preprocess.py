import cv2
import os
import shutil
import tqdm

import numpy as np

from constants import *

def grab_faces(raw_name="raw", faces_name="faces"):
    raw_path = f"{DATA_PATH}/{raw_name}"
    processed_path = f"{DATA_PATH}/{faces_name}"

    if os.path.exists(processed_path):
        shutil.rmtree(processed_path)
    os.makedirs(processed_path)

    face_classifier = cv2.CascadeClassifier(
        cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    )

    fnames = list(os.listdir(raw_path))

    print("Starting face detection . . .")

    for fname in tqdm.tqdm(fnames):
        fpath = f"{raw_path}/{fname}"
        img = cv2.imread(fpath)
        gray_image = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces, reject_levels, level_weights = face_classifier.detectMultiScale3(
            gray_image, scaleFactor=1.1, minNeighbors=10, minSize=(100, 100), outputRejectLevels=1
        )
        i = 0
        for ((x, y, w, h), conf) in zip(faces, level_weights):
            if conf < 5.5:
                continue
            f_img = img[y:y+h, x:x+w, :]
            fwpath = f"{processed_path}/{'.'.join(fname.split('.')[:-1])}_{i}_{conf:0.4f}.jpg"
            cv2.imwrite(fwpath, f_img)
            i += 1

    return processed_path

def resize_faces(faces_name="faces", resized_faces_name="resized"):
    raw_path = f"{DATA_PATH}/{faces_name}"
    processed_path = f"{DATA_PATH}/{resized_faces_name}"

    if os.path.exists(processed_path):
        shutil.rmtree(processed_path)
    os.makedirs(processed_path)

    fnames = list(os.listdir(raw_path))

    print("Starting resizing . . .")

    for fname in tqdm.tqdm(fnames):
        fpath = f"{raw_path}/{fname}"
        img = cv2.imread(fpath)

        f_img = cv2.resize(img, (128, 128))
        fwpath = f"{processed_path}/{fname}"
        cv2.imwrite(fwpath, f_img)
    
    return processed_path

def gray_faces(color_name="resized", gray_name="gray"):
    raw_path = f"{DATA_PATH}/{color_name}"
    processed_path = f"{DATA_PATH}/{gray_name}"

    if os.path.exists(processed_path):
        shutil.rmtree(processed_path)
    os.makedirs(processed_path)

    fnames = list(os.listdir(raw_path))

    print("Starting gray-scaling . . .")

    for fname in tqdm.tqdm(fnames):
        fpath = f"{raw_path}/{fname}"
        img = cv2.imread(fpath)

        f_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        fwpath = f"{processed_path}/{fname}"
        cv2.imwrite(fwpath, f_img)

    return processed_path

def create_numpy_array(name="gray", save_name="dataset"):
    raw_path = f"{DATA_PATH}/{name}"
    save_path = f"{DATA_PATH}/{save_name}.npy"

    fnames = list(os.listdir(raw_path))

    print("Loading into numpy array")

    imgs = []

    for fname in tqdm.tqdm(fnames):
        fpath = f"{raw_path}/{fname}"
        img = cv2.imread(fpath)
        f_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        imgs.append(f_img)

    print("Converting to numpy array . . .")

    imgs = np.array(imgs)

    np.save(save_path, imgs)

    return imgs

def load_dataset(save_name="dataset"):
    save_path = f"{DATA_PATH}/{save_name}.npy"
    return np.load(save_path)

def main():
    # grab_faces()
    # resize_faces()
    # gray_faces()
    # create_numpy_array()

    imgs = load_dataset()
    print(imgs.shape)

    pass

if __name__ == "__main__":
    main()