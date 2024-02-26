import re
import string
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer

class Preprocessor(object):
    def __init__(self, config, logger):
        self.config = config
        self.logger = logger
        self.classes = self.config['classes']  #such as toxic, obscene
        self._load_data()

    @staticmethod
    def clean_text(text):
        if isinstance(text, str):
            text = text.strip().lower().replace('\n', '')
            # tokenization
            words = re.split(r'\W+', text)  # or just words = text.split()
            # filter punctuation
            filter_table = str.maketrans('', '', string.punctuation)
            clean_words = [w.translate(filter_table) for w in words if len(w.translate(filter_table))]
            return clean_words
        else:
            return []


    def _parse(self, data_frame, is_test = False):
        X = data_frame[self.config['input_text_column']].apply(Preprocessor.clean_text).values
        Y = None
        if not is_test:
            columns_to_drop = [self.config['input_id_column'], self.config['input_text_column']]
            Y = data_frame.drop(columns=columns_to_drop, axis=1).values
        else:
            Y = data_frame.Test_ID.values
        return X, Y

    def _load_data(self):

        # define train dataset
        data_df = pd.read_csv(self.config['input_trainset'])
        # Seperate X and Y in train dataset
        self.data_x, self.data_y = self._parse(data_df)
        self.train_x, self.validate_x, self.train_y, self.validate_y = train_test_split(
            self.data_x, self.data_y,
            test_size= self.config['split_ratio'],
            random_state=self.config['random_seed'])

        # define test dataset
        test_df = pd.read_csv(self.config['input_testset'])
        self.test_x, self.test_ids = self._parse(test_df, is_test=True)

    def process(self):
        input_convertor = self.config.get('input_convertor', None)  #指定输入方案

        data_x, data_y, train_x, train_y, validate_x, validate_y, test_x = \
            self.data_x, self.data_y, \
            self.train_x, self.train_y, \
            self.validate_x, self.validate_y, self.test_x

        if input_convertor == 'count_vectorization':
            train_x, validate_x, test_x = self.count_vectorization(train_x, validate_x, test_x)

        return data_x, data_y, train_x, train_y, validate_x, validate_y, test_x

    def count_vectorization(self, train_x, validate_x, test_x):
        vectorizer = CountVectorizer(tokenizer=lambda x:x, preprocessor=lambda x:x)
        vectorized_train_x = vectorizer.fit_transform(train_x)
        vectorized_validate_x = vectorizer.transform(validate_x)
        vectorized_test_x  = vectorizer.transform(test_x)
        return vectorized_train_x, vectorized_validate_x, vectorized_test_x