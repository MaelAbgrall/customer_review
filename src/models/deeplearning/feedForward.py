# python integrated

# libraries
import tensorflow
import keras
from keras import backend as K
K.tensorflow_backend._get_available_gpus()
from keras import callbacks

# project


class feedForward():
    def __init__(self, input_shape, epochs=1000, batch_size=10, debug=0):
        self.batch_size = batch_size
        # a callback is here to automatically stop training
        self.epochs = epochs
        self.debug = debug

        self.callbacks = [
            callbacks.EarlyStopping(min_delta=0.1, patience=20, verbose=1),
            callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, verbose=1, min_delta=0.01, cooldown=0, min_lr=0)
        ]

        self._construct_model(input_shape)

    def _construct_model(self, input_shape):
        hidden_size = int(input_shape * 0.5)

        input_layer = keras.layers.Input(shape=(input_shape, ))
        hidden_layer1 = keras.layers.Dense(input_shape)(input_layer)
        hidden_layer2 = keras.layers.Dense(hidden_size)(hidden_layer1)
        output_layer = keras.layers.Dense(
            1, activation='softmax')(hidden_layer2)

        self.model = keras.models.Model(
            inputs=input_layer, outputs=output_layer)

        self.model.compile(loss='binary_crossentropy',
                           optimizer='adam', metrics=['accuracy'])

    def fit(self, train_features, train_labels, validation_features, validation_labels):
        validation_data = (validation_features, validation_labels)

        fit_history = self.model.fit(train_features,
            train_labels,
            batch_size=self.batch_size,
            epochs=self.epochs,
            verbose=self.debug,
            callbacks=self.callbacks,
            validation_data=validation_data)
        return fit_history.history

    def predict(self, features):
        return self.model.predict(features, batch_size=self.batch_size)

    def evaluate(self, features, labels):
        return self.model.evaluate(features, labels, batch_size=self.batch_size)
