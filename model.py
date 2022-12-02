import recognition


def create_face_recognition():
    print('\nStarting the app...')
    recognition.setup()

    print('\nNow the model will be trained based on your pictures...')
    train_data, test_data = recognition.create_dataset()

    siamese_network = recognition.build_siamese_network_model()
    recognition.run_siamese_network_training(
        train_data=train_data,
        test_data=test_data,
        model=siamese_network,
        n_epochs=25
    )


if __name__ == '__main__':
    create_face_recognition()
