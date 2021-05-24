from 预处理 import pre_processing_df, cut_by_feature_and_interval, build_labels
import pandas as pd
from sklearn.svm import SVR
from sklearn.metrics import mean_absolute_error

from sklearn.naive_bayes import MultinomialNB
from sklearn.neural_network import MLPClassifier
from sklearn import tree
from sklearn.metrics import classification_report

file_name = 'sh600000.csv'
build_labels(file_name)


# 预测均价
def predict_by_SVM(train_x, train_y, test_x):
    # input
    new_train_x = pre_processing_df(train_x).iloc[:, 1:]
    # open, high, low, close, volume, outstanding, turnover
    new_train_x = cut_by_feature_and_interval(new_train_x, ['open', 'close'], 40)
    new_train_y = train_y.iloc[60:].iloc[:, 1:]
    # output compare
    new_test_x = pre_processing_df(test_x).iloc[:, 1:]
    new_test_x = cut_by_feature_and_interval(new_test_x, ['open', 'close'], 40)

    svm_model = SVR()
    svm_model.set_params(**{'kernel': "poly", 'degree': 1, 'C': 15000,
                            'gamma': 'scale', 'coef0': 0.0, 'tol': 0.001, 'epsilon': 1})

    model = svm_model.fit(new_train_x, new_train_y)
    predict_test_Y = model.predict(new_test_x)
    return predict_test_Y


def compare_by_MAE(predict_y):
    truth_test_Y = pd.read_csv('sh600000_y_test_avg_price.csv', sep=',', index_col=0)['Avg_price'].to_list()[60:]
    print('truth_Y\n', truth_test_Y, '\n')
    MeanAbsError = mean_absolute_error(predict_y, truth_test_Y)
    print('MeanAbsError = ', MeanAbsError)


train_X = pd.read_csv('sh600000_x_train.csv', sep=',', index_col=0)
# train_y = pd.read_csv('sh600000_y_train_increase_decrease.csv', sep=',', index_col=0)
train_Y = pd.read_csv('sh600000_y_train_avg_price.csv', sep=',', index_col=0)
# print(train_x)
test_X = pd.read_csv('sh600000_x_test.csv', sep=',', index_col=0)
# test_y = pd.read_csv('sh600000_y_test_increase_decrease.csv', sep=',', index_col=0)
test_Y = pd.read_csv('sh600000_y_test_avg_price.csv', sep=',', index_col=0)

predict_Y = predict_by_SVM(train_X, train_Y, test_X)
print('predict_Y', predict_Y)
compare_by_MAE(predict_Y)

'''''''''

# Deccision Tree
classifier_DT = tree.DecisionTreeClassifier(criterion="entropy")
model_DT = classifier_DT.fit(new_train_x, new_train_y)
Y_test_predict = model_DT.predict(new_test_x)
print("Deccision Tree:\n")
print(classification_report(new_test_y, Y_test_predict))

# MultinomialNB
classifier_MNB = MultinomialNB()
model_MNB = classifier_MNB.fit(new_train_x, new_train_y)
Y_test_predict = model_MNB.predict(new_test_x)
print("Multinomial Naive Bayes:\n")
print(classification_report(new_test_y, Y_test_predict))

# Artificial Neural Networks (ANN)
# MLPClassifier
classifier_ANN = MLPClassifier(hidden_layer_sizes=(200,))
model_ANN = classifier_ANN.fit(new_train_x, new_train_y)
Y_test_predict = model_ANN.predict(new_test_x)
print("Artificial Neural Networks:\n")
print(classification_report(new_test_y, Y_test_predict))
'''''''''
