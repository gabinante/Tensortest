import matplotlib.pyplot as plt
import numpy as np
import PIL
import tensorflow as tf

from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential
import pathlib
import logging
import click


def classify():
    # First, ingest a given dataset.
    training_set = ingest_dataset()
    # next, build the model if the flag is selected. Otherwise we will use the tflite model.
    if build == True:
        image_model = build_model(training_set)
    # Then, convert the model to a tflite model unless we are in training mode.
    if not training:
        image_model = convert_model(image_model)
    # Now, test a given image against our trained model
    results = test_image(image_path, image_model)
    # See how we did
    publish_results(results)


def ingest_dataset():
    if custom_dataset == False:
        dataset_url = "https://storage.googleapis.com/download.tensorflow.org/example_images/flower_photos.tgz"
        data_dir = tf.keras.utils.get_file('flower_photos.tar', origin=dataset_url, extract=True)
    else:
        data_dir = tf.keras.utils.image_dataset_from_directory(custom_dataset)

    data_dir = pathlib.Path(data_dir).with_suffix('')

    image_count = len(list(data_dir.glob('*/*.jpg')))
    classification_count = len(list(data_dir.glob('*')))
    logging.debug(f'training set includes {image_count} images spanning {classification_count} classifications')
    return data_dir

def build_model(training_set):
    ''' Construct a tensorflow model from scratch using a specified training set '''

    # loader parameters
    batch_size = 32
    img_height = 180
    img_width = 180

    # train using 80% of the images
    train_ds = tf.keras.utils.image_dataset_from_directory(
        data_dir,
        validation_split=0.2,
        subset="training",
        seed=123,
        image_size=(img_height, img_width),
        batch_size=batch_size)

    # validation set using 20% of the images
    val_ds = tf.keras.utils.image_dataset_from_directory(
        data_dir,
        validation_split=0.2,
        subset="validation",
        seed=123,
        image_size=(img_height, img_width),
        batch_size=batch_size)

    # All potential classes
    class_names = train_ds.class_names
    print(class_names)

    plt.figure(figsize=(10, 10))
    for images, labels in train_ds.take(1):
        for i in range(9):
            ax = plt.subplot(3, 3, i + 1)
            plt.imshow(images[i].numpy().astype("uint8"))
            plt.title(class_names[labels[i]])
            plt.axis("off")

    AUTOTUNE = tf.data.AUTOTUNE

    train_ds = train_ds.cache().shuffle(1000).prefetch(buffer_size=AUTOTUNE)
    val_ds = val_ds.cache().prefetch(buffer_size=AUTOTUNE)

    normalization_layer = layers.Rescaling(1./255)

    num_classes = len(class_names)

    # Data augmentation to solve for overfitting
    data_augmentation = keras.Sequential(
      [
        layers.RandomFlip("horizontal",
                          input_shape=(img_height,
                                      img_width,
                                      3)),
        layers.RandomRotation(0.1),
        layers.RandomZoom(0.1),
      ]
    )

    # # Print a visual representation of the model
    # plt.figure(figsize=(10, 10))
    # for images, _ in train_ds.take(1):
    #   for i in range(9):
    #     augmented_images = data_augmentation(images)
    #     ax = plt.subplot(3, 3, i + 1)
    #     plt.imshow(augmented_images[0].numpy().astype("uint8"))
    #     plt.axis("off")

    # RGB uses 255 channels which is suboptimal. use keras Rescaling to normalize.
    model = Sequential([
      data_augmentation,
      layers.Rescaling(1./255),
      layers.Conv2D(16, 3, padding='same', activation='relu'),
      layers.MaxPooling2D(),
      layers.Conv2D(32, 3, padding='same', activation='relu'),
      layers.MaxPooling2D(),
      layers.Conv2D(64, 3, padding='same', activation='relu'),
      layers.MaxPooling2D(),
      layers.Dropout(0.2),
      layers.Flatten(),
      layers.Dense(128, activation='relu'),
      layers.Dense(num_classes, name="outputs")
    ])
    logger.debug("Compiling model...")
    model.compile(optimizer='adam',
                  loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                  metrics=['accuracy'])

    # Show all the model layers
    model.summary()

    # train for 15 epochs
    epochs = 15
    history = model.fit(
      train_ds,
      validation_data=val_ds,
      epochs=epochs
    )

    acc = history.history['accuracy']
    val_acc = history.history['val_accuracy']

    loss = history.history['loss']
    val_loss = history.history['val_loss']

    epochs_range = range(epochs)

    plt.figure(figsize=(8, 8))
    plt.subplot(1, 2, 1)
    plt.plot(epochs_range, acc, label='Training Accuracy')
    plt.plot(epochs_range, val_acc, label='Validation Accuracy')
    plt.legend(loc='lower right')
    plt.title('Training and Validation Accuracy')

    plt.subplot(1, 2, 2)
    plt.plot(epochs_range, loss, label='Training Loss')
    plt.plot(epochs_range, val_loss, label='Validation Loss')
    plt.legend(loc='upper right')
    plt.title('Training and Validation Loss')
    plt.show()


def test_image(image_path, model):
    img = tf.keras.utils.load_img(
        image_path, target_size=(img_height, img_width)
    )
    img_array = tf.keras.utils.img_to_array(img)
    img_array = tf.expand_dims(img_array, 0) # Create a batch

    if training:
        # test using keras model
        predictions = model.predict(img_array)
        score = tf.nn.softmax(predictions[0])

        print(
            "This image most likely belongs to {} with a {:.2f} percent confidence."
            .format(class_names[np.argmax(score)], 100 * np.max(score))
        )
    else:
        interpreter = tf.lite.Interpreter(model_path=tflite_model_path)

        interpreter.get_signature_list()
        classify_lite = interpreter.get_signature_runner('serving_default')
        classify_lite
        predictions_lite = classify_lite(sequential_1_input=img_array)['outputs']
        score_lite = tf.nn.softmax(predictions_lite)
        print(
            "This image most likely belongs to {} with a {:.2f} percent confidence."
            .format(class_names[np.argmax(score_lite)], 100 * np.max(score_lite))
        )

        print(np.max(np.abs(predictions - predictions_lite)))

def convert_model(model):
    # Convert the model.
    logger.debug("converting to tflite model")
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    tflite_model = converter.convert()

    # Save the model.
    with open('model.tflite', 'wb') as f:
      f.write(tflite_model)


if __name__ == '__main__':
    logger=logging.getLogger()
    @click.command()
    @click.option('--build', default=False, help='Choose whether to build the model from scratch. If this option is not selected, we will use the tflite model.')
    @click.option('--training', default=False, help='do not convert the model to tflite, to save time between iterations')
    @click.option('--tflite_model_path', default='model.tflite', help='a path to a custom tflite model')
    @click.option('--image', default='dataset/sunflower_test.jpg', help='an image to test the model against')
    @click.option('--dataset_directory', default=False, help='Specify a directory from which to retrieve a dataset')

    logger.debug('lets go!')
    classify()
