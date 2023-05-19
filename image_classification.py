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
import yaml
from pathlib import Path

@click.command()
@click.option('--build', default=True, help='Choose whether to build the model from scratch. If this option is not selected, we will use the tflite model.')
@click.option('--convert', default=True, help='Convert the model to tflite. (True/False)')
@click.option('--tflite_model_path', default='model.tflite', help='a path to a custom tflite model')
@click.option('--image_path', default='dataset/sunflower_test.jpg', help='an image to test the model against')
@click.option('--dataset_directory', default=False, help='Specify a directory from which to retrieve a dataset')
@click.option('--num_epochs', default=10, help='customize the number of training iterations')
def classify(build, convert, tflite_model_path, image_path, dataset_directory, num_epochs):
    # First, build the model if the flag is selected. Otherwise we will use the tflite model.
    # if build is false and convert is true
    # True True √
    # False True can't build a model out of nothing
    # False False use existing model
    # True False training mode for fast iterations
    if build == True:
        keras_model, class_names = build_model(dataset_directory, num_epochs)
    # Then, convert the model to a tflite model unless we are in training mode.
        if convert == True:
            tflite_model = convert_model(keras_model)
    elif build == False and convert == False:
        keras_model = None
        tflite_model = tflite_model_path
        class_names = None
    # Now, test a given image against our trained model
    results = test_image(image_path, keras_model, tflite_model, build, convert, tflite_model_path, class_names)
    # See how we did
    # publish_results(results)

def build_model(custom_dataset, num_epochs):
    ''' Construct a tensorflow model from scratch using a specified training set '''

    if custom_dataset == False:
        dataset_url = "https://storage.googleapis.com/download.tensorflow.org/example_images/flower_photos.tgz"
        data_dir = tf.keras.utils.get_file('flower_photos.tar', origin=dataset_url, extract=True)
    else:
        data_dir = tf.keras.utils.image_dataset_from_directory(custom_dataset)

    data_dir = pathlib.Path(data_dir).with_suffix('')

    image_count = len(list(data_dir.glob('*/*.jpg')))
    classification_count = len(list(data_dir.glob('*')))
    logging.debug(f'training set includes {image_count} images spanning {classification_count} classifications')

    # loader parameters. we should probably not hard code these
    batch_size = 16
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

    
    params = yaml.safe_load(Path('classifier_params.yaml').read_text())
    print(params)

    # RGB uses 255 channels which is suboptimal. use keras Rescaling to normalize.
    # Tweaking to add additional convolutional and fully connected layers.
    # We move up our kernel / filters over time to first capture details and Then
    # gather additional macro details in deeper layers.
    model = Sequential([
        data_augmentation,  # Data augmentation to generate additional training samples
        layers.Rescaling(1./255),  # Rescale pixel values from [0, 255] to [0, 1]
        layers.Conv2D(params['convolution_1']['value'], (params['convolution_1']['filter_1'], params['convolution_1']['filter_2']), activation='relu', input_shape=(180, 180, 3)),  # x filters with a 3x3 kernel
        layers.MaxPooling2D((3, 3)),  # Max pooling with a 3x3 pool size
        layers.Conv2D(params['convolution_2']['value'], (params['convolution_2']['filter_1'], params['convolution_2']['filter_2']), activation='relu'),  # x filters with a 4x4 kernel
        layers.MaxPooling2D(),  # Default pool size (2x2) for the previous layer output
        layers.Conv2D(params['convolution_3']['value'], (params['convolution_3']['filter_1'], params['convolution_3']['filter_2']), activation='relu'),  # x filters with a 6x6 kernel, ReLU activation
        layers.MaxPooling2D((2, 2)),  # Max pooling with a 2x2 pool size
        layers.Dropout(params['dropout']),  # Dropout layer with a 20% dropout rate to reduce overfitting
        layers.Flatten(),  # Flatten the output from the previous layer
        layers.Dense(params['fully_connected_1'], activation='relu'),  # Fully connected layer with x units, ReLU activation
        layers.Dense(num_classes, name="outputs")  # Output layer with num_classes units, representing the predicted class probabilities
    ])
    logger.debug("Compiling model...")
    model.compile(optimizer='adam',
                  loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                  metrics=['accuracy'])

    # Show all the model layers
    model.summary()

    history = model.fit(
      train_ds,
      validation_data=val_ds,
      epochs=num_epochs
    )

    acc = history.history['accuracy']
    val_acc = history.history['val_accuracy']

    loss = history.history['loss']
    val_loss = history.history['val_loss']

    epochs_range = range(num_epochs)

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

    return model, class_names

def test_image(image_path, keras_model, tflite_model, build, convert, tflite_model_path, class_names):
    batch_size = 16
    img_height = 180
    img_width = 180
    img = tf.keras.utils.load_img(
        image_path, target_size=(img_height, img_width)
    )
    img_array = tf.keras.utils.img_to_array(img)
    img_array = tf.expand_dims(img_array, 0) # Create a batch



    if build:
        # test using keras model
        predictions = keras_model.predict(img_array)
        score = tf.nn.softmax(predictions[0])

        print(
            "This image most likely belongs to {} with a {:.2f} percent confidence."
            .format(class_names[np.argmax(score)], 100 * np.max(score))
        )
    if convert == True or build == False and convert == False:
        interpreter = tf.lite.Interpreter(model_path=tflite_model_path)

        signature_list = interpreter.get_signature_list()
        logger.debug(f'signature list: {signature_list}')
        signature_name = list(signature_list.keys())[0]
        logger.debug(signature_name)
        classify_lite = interpreter.get_signature_runner(f'{signature_name}')

        predictions_lite = classify_lite(sequential_input=img_array)['outputs']
        score_lite = tf.nn.softmax(predictions_lite)
        try:
            model_meta = interpreter.get_tensor_details()[0]['metadata']
            meta_dict = model_meta.get('metadata')
            if meta_dict is not None:
                if 'class_names' in meta_dict:
                    class_names = meta_dict['class_names'].decode('utf-8').split('\n')
                    print(class_names)
        except Exception as e:
            logger.debug('failed to find class names, defaulting to hand-jam')
            class_names = ['daisy', 'dandelion', 'roses', 'sunflowers', 'tulips']
        print(
            "This image most likely belongs to {} with a {:.2f} percent confidence."
            .format(class_names[np.argmax(score_lite)], 100 * np.max(score_lite))
        )

        # # calculate the difference between the models
        # print(np.max(np.abs(predictions - predictions_lite)))

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
    classify()
