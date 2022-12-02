import os

import cv2
import numpy as np
import tensorflow as tf
from kivy.app import App
from kivy.clock import Clock
from kivy.graphics.texture import Texture
from kivy.logger import Logger
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.button import Button
from kivy.uix.image import Image
from kivy.uix.label import Label

import recognition


class FacialVerificationApp(App):
    def build(self):
        self.video = Image(size_hint=(1, 0.8))
        self.label = Label(text='', size_hint=(1, 0.1))
        self.button = Button(
            text='Verify User', on_press=self.verify_user, size_hint=(1, 0.1)
        )

        layout = BoxLayout(orientation='vertical')
        layout.add_widget(self.video)
        layout.add_widget(self.button)
        layout.add_widget(self.label)

        self.model = tf.keras.models.load_model(
            'siamese_network.h5',
            custom_objects={'L1DistanceLayer': recognition.L1DistanceLayer},
        )

        self.capture = cv2.VideoCapture(0)
        Clock.schedule_interval(self.update_video, 1.0 / 33.0)

        return layout

    def update_video(self, *args):
        _, frame = self.capture.read()
        frame = frame[120:370, 200:450, :]
        buffer = cv2.flip(frame, 0).tostring()

        texture = Texture.create(
            size=(frame.shape[1], frame.shape[0]),
            colorfmt='bgr'
        )

        texture.blit_buffer(buffer, colorfmt='bgr', bufferfmt='ubyte')
        self.video.texture = texture

    def preprocess_image(self, path):
        return recognition.preprocess_image(path)

    def verify_user(self, *args):
        _, frame = self.capture.read()
        frame = frame[120 : 120 + 250, 200 : 200 + 250, :]
        cv2.imwrite(recognition.APP_INPUT_IMAGE_PATH, frame)

        results = []
        verification_images = os.listdir(recognition.APP_VERIFICATION_PATH)

        for image in verification_images:
            input = self.preprocess_image(recognition.APP_INPUT_IMAGE_PATH)
            user_image = os.path.join(recognition.APP_VERIFICATION_PATH, image)
            user_image = self.preprocess_image(user_image)

            comparison_data = list(np.expand_dims([input, user_image], axis=1))
            results.append(self.model.predict(comparison_data))

        detection = np.sum(np.array(results) > recognition.DETECTION_THRESHOLD)
        verification = detection / len(verification_images)
        verified = verification > recognition.VERIFICATION_THRESHOLD

        Logger.info(f'{round(verification * 100, 2)}% match!')
        self.label.text = 'Verified' if verified else 'Unverified'

        return results, verified


if __name__ == '__main__':
    FacialVerificationApp().run()
