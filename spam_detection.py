import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer as cv
from sklearn.model_selection import train_test_split as tts
from sklearn.naive_bayes import MultinomialNB as mnb

# import data using pandas
data = pd.read_table('SMSSpamCollection', sep='\t', header=None, names=['label', 'sms_message'])

# map label to binary
data['label'] = data.label.map({'ham':0, 'spam':1})

# print the first 5 columns
# print (data.head(n=5))

x_train , x_test, y_train, y_test = tts(data['sms_message'], data['label'], random_state=1)

# print('Number of rows in the total set: {}'.format(data.shape[0]))
# print('Number of rows in the training set: {}'.format(x_train.shape[0]))
# print('Number of rows in the test set: {}'.format(x_test.shape[0]))

count_vector = cv()
training_data = count_vector.fit_transform(x_train)
testing_data = count_vector.transform(x_test)

naive_bayes = mnb()
naive_bayes.fit(training_data, y_train)

predictions = naive_bayes.predict(testing_data)

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
print('Accuracy score: ', format(accuracy_score(y_test, predictions)))
print('Precision score: ', format(precision_score(y_test, predictions)))
print('Recall score: ', format(recall_score(y_test, predictions)))
print('F1 score: ', format(f1_score(y_test, predictions)))
