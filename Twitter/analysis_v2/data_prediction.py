import joblib
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report


from Twitter.analysis_v2.data_prediction_ml import get_test_train_data, prepare_model_data, split_data


def predict():
    train_df, test_df = get_test_train_data(False)

    train_df = prepare_model_data(train_df)
    test_df = prepare_model_data(test_df)

    X_train, y_train, X_test, y_test = split_data(train_df, test_df)

    popularity_model = joblib.load('../../data/models/popularity.joblib')
    predictions = popularity_model.predict(X_test)
    print(accuracy_score(y_test, predictions))
    print(confusion_matrix(y_test, predictions))
    print(classification_report(y_test, predictions))


if __name__ == '__main__':
    predict()