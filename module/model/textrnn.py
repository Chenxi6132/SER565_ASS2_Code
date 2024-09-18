from keras.models import Sequential
from keras.optimizers import Adam
from keras.layers import Embedding, SimpleRNN, Dense

class TextRNN(object):
    def __init__(self, classes, config):
        self.classes = classes
        self.config = config
        self.num_classes = len(classes)

        self.model = self._build()

    def _build(self):
        model = Sequential()
        model.add(Embedding(self.config['vocab_size'], self.config['embedding_dim'], input_length=self.config['maxlen'], trainable=True))
        model.add(SimpleRNN(self.config['rnn_units']))  # Use optimized RNN units
        model.add(Dense(self.num_classes, activation='sigmoid'))
        optimizer = Adam(learning_rate=self.config['learning_rate'])  # dynamic learning rate

        model.compile(optimizer=optimizer, loss='BinaryCrossentropy', metrics='accuracy')
        model.summary()
        return model

    def fit(self, train_x, train_y):
        self.model.fit(train_x, train_y, epochs=self.config['epochs'], verbose=True, batch_size=self.config['batch_size'])

    def predict(self, test_x):
        probs = self.model.predict(test_x)
        return probs >= 0.5

    def predict_prob(self, test_x):
        return self.model.predict(test_x)


