import csv


class Predictor(object):
    def __init__(self, config, logger, model):
        self.config = config
        self.logger = logger
        self.model = model

    def predict(self, test_x):
        predictions = self.model.predict(test_x)
        return predictions

    def predict_prob(self, test_x):
        predictions = self.model.predict_prob(test_x)
        return predictions

    def save_result(self, test_ids, probs):
        with open(self.config['output_path'], 'w') as output_csv_file:
            header= ['Descrption', 'Test_ID', 'Functionality_Test_Case', 'User_Interface_Test_Case', 'Performance_Test_Case'\
                     'Integration_Test_Case', 'Usability_Test_Case', 'Database_Test_Case', 'Security_Test_Case', 'User_Acceptance_Test_Case']

            writer = csv.writer(output_csv_file)
            writer.writerow(header)
            for test_id, prob in zip(test_ids, probs.tolist()):
                writer.writerow([test_id] + prob)