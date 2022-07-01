import math
from os import walk

import joblib
import numpy as np
import pandas as pd
import datetime
from sklearn.metrics import mean_absolute_error, mean_squared_error
import plotly.express as px
from sklearn.ensemble import GradientBoostingRegressor
import plotly


num_folds = 7
seed = 7
scoring = 'accuracy'
validation_size = 0.70
BASE_FOLDER = "../../data/processed_tweets/"
vars = ['like_count', 'retweet_count', 'quote_count', 'reply_count', 'reach', 'topics_ids', 'sentiment_enc', 'day_phase_enc', 'day_of_week_enc', 'month_enc', 'popularity', 'followers', 'following', 'tweet_count', 'verified_enc', 'seniority']
tweet_vars = ['like_count', 'retweet_count', 'quote_count', 'reply_count', 'reach', 'sentiment_enc', 'day_phase_enc', 'day_of_week_enc', 'month_enc', 'topics_ids']
users_vars = ['followers', 'following', 'tweet_count', 'verified_enc', 'seniority']
num_vars = ['like_count', 'retweet_count', 'quote_count', 'reply_count', 'reach', 'followers', 'following', 'tweet_count', 'seniority']
cat_vars = ['sentiment_enc', 'verified_enc', 'day_of_week_enc', 'day_phase_enc', 'month_enc']
variables_to_predict = ['followers', 'following', 'tweet_count', 'seniority', 'verified_enc', 'day_phase_enc', 'day_of_week_enc', 'month_enc', 'topics_ids', 'sentiment_enc', 'timestamp', 'retweet_count', 'popularity', 'year']
variables_to_keep = ['followers', 'following', 'tweet_count', 'seniority', 'verified_enc', 'day_phase_enc', 'day_of_week_enc', 'month_enc', 'topics_ids', 'sentiment_enc', 'timestamp', 'retweet_count']


def predict():
    train_df, test_df = get_test_train_data()

    train_df = prepare_model_data(train_df)
    test_df = prepare_model_data(test_df)

    X_train, y_train, X_test, y_test = split_data(train_df, test_df)
    train_time_series_with_folds(X_train, y_train, X_test, y_test)


def get_test_train_data():
    filenames = next(walk(BASE_FOLDER), (None, None, []))[2]
    print(filenames)

    if len(filenames) == 2:
        train_df = pd.read_csv(filepath_or_buffer=BASE_FOLDER + filenames[0], sep=",")
        test_df = pd.read_csv(filepath_or_buffer=BASE_FOLDER + filenames[1], sep=",")
    elif len(filenames) > 2:
        test_df = pd.read_csv(filepath_or_buffer=BASE_FOLDER + filenames[len(filenames) - 1], sep=",")
        test_df = convert_and_sort_time(test_df)

        train_df = pd.DataFrame()
        for i in range(len(filenames) - 1):
            df_temp = pd.read_csv(filepath_or_buffer=BASE_FOLDER + filenames[i], sep=",")
            df_temp = convert_and_sort_time(df_temp)
            train_df = pd.concat([train_df, pd.DataFrame.from_records(df_temp)])

    return train_df, test_df


def convert_and_sort_time(df):
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df['timestamp'] = [i.replace(tzinfo=datetime.timezone.utc) for i in df['timestamp']]
    return df.sort_values(by='timestamp', ascending=True)


def prepare_model_data(df):
    df = df[(df['topics_ids'] != -1) & (df['topics'] == 'Person')]
    df = df[variables_to_predict].resample('D', on='timestamp').mean()
    return df.dropna(how='any')


def split_data(train_df, test_df):
    X_train = train_df.drop('retweet_count', axis=1)
    y_train = train_df['retweet_count']
    print(X_train.shape)
    print(y_train.shape)
    X_test = test_df.drop('retweet_count', axis=1)
    y_test = test_df['retweet_count']
    print(X_test.shape)
    print(y_test.shape)
    return X_train, y_train, X_test, y_test

def train_time_series_with_folds(X_train, y_train, X_test, y_test):
    model = GradientBoostingRegressor(random_state=42)
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)
    joblib.dump(model, '../../data/models/topic_performance.joblib')

    mae = np.round(mean_absolute_error(y_test, predictions), 3)
    mse = np.round(mean_squared_error(y_test, predictions), 3)
    rmse = np.round(math.sqrt(mse), 3)
    print(mae)
    print(mse)
    print(rmse)

    draw_results_charts(model, X_test.reset_index()['timestamp'], y_test, predictions, mae)


def draw_results_charts(model, timestamps, test_targets, predictions, mae):
    res = []
    for i in range(365):
        d = dict()
        d['timestamp'] = timestamps[i]
        d['type'] = 'real'
        d['value'] = test_targets[i]
        res.append(d)

        d = dict()
        d['timestamp'] = timestamps[i]
        d['type'] = 'prediction'
        d['value'] = predictions[i]
        res.append(d)

    df_results = pd.DataFrame(res)

    fig = px.line(df_results, x="timestamp", y="value", color='type',
                  title=("Predictions of topic retweet count average for 2021 with MAE:" + str(mae)),
                  color_discrete_sequence=px.colors.qualitative.Safe, width=900, height=500)
    # fig.add_trace()
    fig.update_xaxes(title_text="Time")
    fig.update_yaxes(title_text="Average retweet count")
    fig.show()
    plotly.offline.plot(fig,
                        filename='../../data/charts/Predictions of topic retweet count average for 2021 with MAE.html')

    # create a dataframe with the variable importances of the model
    df_importances = pd.DataFrame({
        'feature': model.feature_names_in_,
        'importance': model.feature_importances_
    }).sort_values(by='importance', ascending=False)

    fig = px.bar(df_importances, x="feature", y="importance", title=("Variable Importances"),
                 color_discrete_sequence=px.colors.qualitative.Safe, width=900, height=500)
    fig.show()


if __name__ == '__main__':
    predict()