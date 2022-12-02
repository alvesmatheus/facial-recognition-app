import random
import shutil
import subprocess
import os
import uuid

import cv2

from recognition.constants import ANCHOR_PATH, NEGATIVE_PATH, POSITIVE_PATH
from recognition.constants import LABELED_FACES_PATH
from recognition.constants import TRAIN_CHECKPOINTS_PATH
from recognition.constants import APP_VERIFICATION_PATH


def create_directories():
    print('Creating directory structure...')
    paths = [ANCHOR_PATH, POSITIVE_PATH, NEGATIVE_PATH, TRAIN_CHECKPOINTS_PATH]
    for p in paths:
        if not os.path.exists(p):
            os.makedirs(p)


def uncompress_negative_data():
    print('Uncompressing Labeled Faces in the Wild dataset...')
    if not os.path.exists(LABELED_FACES_PATH):
        subprocess.run(['tar', '-xf', 'lfw.tgz'])

    for directory in os.listdir(LABELED_FACES_PATH):
        for file in os.listdir(os.path.join(LABELED_FACES_PATH, directory)):
            origin = os.path.join(LABELED_FACES_PATH, directory, file)
            destination = os.path.join(NEGATIVE_PATH, file)

            os.replace(origin, destination)

    shutil.rmtree(LABELED_FACES_PATH)


def collect_pictures(device=0):
    print(
        '''\nNow let's take some pictures of you...
    - First, press the 'A' key for 30 seconds to capture the ANCHOR PICTURES.
    - Then, press the 'P' key for 30 seconds to capture the POSITIVE PICTURES.
    - Finally, close the camera display using the 'Q' key.'''
    )

    video = cv2.VideoCapture(device)
    while video.isOpened():
        _, frame = video.read()
        frame = frame[120:370, 200:450, :]
        cv2.imshow('Picture Collector ', frame)

        key = cv2.waitKey(1)
        if key == ord('a'):
            photo_path = os.path.join(ANCHOR_PATH, f'{uuid.uuid1()}.jpg')
            cv2.imwrite(photo_path, frame)

        elif key == ord('p'):
            photo_path = os.path.join(POSITIVE_PATH, f'{uuid.uuid1()}.jpg')
            cv2.imwrite(photo_path, frame)

        elif key == ord('q'):
            break

    video.release()
    cv2.destroyAllWindows()


def select_verification_files():
    if os.path.exists(APP_VERIFICATION_PATH):
        shutil.rmtree(APP_VERIFICATION_PATH)
    os.makedirs(APP_VERIFICATION_PATH)

    for i in range(50):
        random_image = random.choice(os.listdir(ANCHOR_PATH))
        image_path = os.path.join(ANCHOR_PATH, random_image)
        shutil.copy(image_path, APP_VERIFICATION_PATH)


def setup(device=0):
    create_directories()
    uncompress_negative_data()
    collect_pictures(device)
    select_verification_files()
