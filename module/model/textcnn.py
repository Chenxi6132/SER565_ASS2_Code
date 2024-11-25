from keras.backend import binary_crossentropy, categorical_crossentropy
from keras.models import Sequential
from keras.layers import Embedding, Conv1D, MaxPooling1D, Dense, Flatten, Dropout
from tensorflow.keras.optimizers import Adam

class TextCNN(object):
    def __init__(self, classes, config):
        self.classes = classes
        self.num_class = len(classes)
        self.config = config
        self.model = self._build()

    def _build(self):
        model = Sequential()
        model.add(Embedding(self.config['vocab_size'], self.config['embedding_dim'], input_length=self.config['maxlen'], trainable=True))
        model.add(Conv1D(self.config['filters_1'], self.config['cnn_kernel_size'], activation='relu', padding='same'))
        model.add(MaxPooling1D())
        model.add(Conv1D(self.config['filters_2'], self.config['cnn_kernel_size'], activation='relu', padding='same'))
        model.add(MaxPooling1D())
        model.add(Conv1D(self.config['filters_3'], self.config['cnn_kernel_size'], activation='relu', padding='same'))
        model.add(MaxPooling1D())
        model.add(Flatten())
        model.add(Dense(128, activation='relu'))
        model.add(Dropout(self.config['dropout_rate']))
        model.add(Dense(self.num_class, activation='sigmoid'))
        optimizer = Adam(learning_rate=self.config['learning_rate'])
        model.compile(optimizer=optimizer, loss='binary_crossentropy', metrics='accuracy')
        model.summary()
        return model

    def fit(self, train_x, train_y):
        self.model.fit(train_x, train_y, epochs=self.config['epochs'], verbose=True, batch_size=self.config['batch_size'])

    def predict(self, test_x):
        probs = self.model.predict(test_x)
        return probs >= 0.5

    def predict_prob(self, test_x):
        return self.model.predict(test_x)



