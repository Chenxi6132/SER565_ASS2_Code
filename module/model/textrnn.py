from keras.models import Sequential
from keras.optimizers import Adam
from keras.layers import Embedding, LSTM, Dense, Bidirectional, Dropout


class TextRNN(object):
    def __init__(self, classes, config):
        self.classes = classes
        self.config = config
        self.num_classes = len(classes)
        self.model = self._build()


    def _build(self):
        model = Sequential()
        model.add(Embedding(self.config['vocab_size'], self.config['embedding_dim'], input_length=self.config['maxlen'], trainable=True))
        model.add(Bidirectional(LSTM(self.config['rnn_units'], return_sequences=False)))  # Use optimized RNN units
        model.add(Dropout(self.config['dropout_rate']))
        # mutli-class sigle label classification use softmax ,categorical_crossentropy
        # multi-class multi-label classification use sigmoid, binary_crossentropy
        model.add(Dense(self.num_classes, activation='softmax'))
        optimizer = Adam(learning_rate=self.config['learning_rate'])  # dynamic learning rate
        model.compile(optimizer = optimizer, loss='binary_crossentropy', metrics='accuracy')
        model.summary()
        return model

    def fit(self, train_x, train_y):
        self.model.fit(train_x, train_y, epochs=self.config['epochs'], verbose=True, batch_size=self.config['batch_size'])

    def predict(self, test_x):
        probs = self.model.predict(test_x)
        return probs >= 0.5

    def predict_prob(self, test_x):
        return self.model.predict(test_x)


