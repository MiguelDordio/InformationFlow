from os import walk

import joblib
import numpy as np
import pandas as pd
import datetime
from sklearn.metrics import mean_absolute_error
import plotly.express as px
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.preprocessing import StandardScaler
import plotly


num_folds = 7
seed = 7
scoring = 'accuracy'
validation_size = 0.70
BASE_FOLDER = "../../data/processed_data"
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
    train_time_series_with_folds(X_train, X_test, y_train, y_test)


def get_test_train_data():
    filenames = next(walk(BASE_FOLDER), (None, None, []))[2]
    print(filenames)

    if len(filenames) == 2:
        train_df = pd.read_csv(filepath_or_buffer=filenames[0], sep=",")
        test_df = pd.read_csv(filepath_or_buffer=filenames[1], sep=",")
    elif len(filenames) > 2:
        train_df = pd.read_csv(filepath_or_buffer=filenames[0], sep=",")
        test_df = pd.read_csv(filepath_or_buffer=filenames[len(filenames) - 1], sep=",")
        for i in range(1, len(filenames) - 1):
            temp_df = pd.read_csv(filepath_or_buffer=filenames[i], sep=",")
            train_df = train_df.append(temp_df, ignore_index=True)

    return train_df, test_df


def prepare_model_data(df):
    df = df[variables_to_predict]
    df = df[(~df['topics'].isnull()) & (df['topics'] != '')]
    return df[variables_to_keep].resample('D', on='timestamp').mean()


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
    #create, train and do inference of the model
    model = GradientBoostingRegressor(random_state=42)
    model.fit(X_train, y_train)

    predictions = model.predict(X_test)

    #calculate MAE
    mae = np.round(mean_absolute_error(y_test, predictions), 3)

    draw_results_charts(model, X_test.index, y_test, predictions, mae)


def draw_results_charts(model, timestamps, test_targets, predictions, mae):
    res = []
    for i in range(365):
        d = dict()
        d['timestamp'] = timestamps[i]
        d['type'] = 'real'
        d['value'] = test_targets
        res.append(d)

        d = dict()
        d['timestamp'] = timestamps
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