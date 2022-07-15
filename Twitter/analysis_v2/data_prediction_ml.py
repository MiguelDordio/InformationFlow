from os import walk

import joblib
import pandas as pd
import datetime
import seaborn as sns
from matplotlib import pyplot as plt
sns.set()
#%%
from sklearn.linear_model import LassoCV
from sklearn.feature_selection import RFE
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from imblearn.over_sampling import SMOTE
from collections import Counter
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier, ExtraTreesClassifier
from sklearn.model_selection import GridSearchCV, KFold, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.feature_selection import chi2
from sklearn.feature_selection import f_classif
from sklearn.feature_selection import SelectKBest


BASE_FOLDER = "../../data/processed_tweets/"

num_vars = ['followers', 'following', 'tweet_count', 'seniority']
cat_vars = ['topics_ids', 'sentiment_enc', 'hashtags_enc', 'verified_enc', 'day_phase_enc', 'day_of_week_enc', 'month_enc']
models = [('LR', LogisticRegression(solver='lbfgs')), ('LDA', LinearDiscriminantAnalysis()),
          ('CART', DecisionTreeClassifier()), ('NB', GaussianNB()), ('AB', AdaBoostClassifier()),
          ('GBM', GradientBoostingClassifier()), ('RFC', RandomForestClassifier(n_estimators=100)),
          ('ET', ExtraTreesClassifier())]
num_folds = 5
seed = 7
scoring = 'accuracy'


def train_test_model():
    train_df, test_df = get_test_train_data(False)

    train_df = prepare_model_data(train_df)
    test_df = prepare_model_data(test_df)

    X_train, y_train, X_test, y_test = split_data(train_df, test_df)
    X_train_cats, X_train_num_scaled, X_test_num_scaled = standardize(X_train, X_test)

    cat_to_keep, num_to_keep = feature_selection(X_train_cats, X_train_num_scaled, y_train)
    X_train, X_test = format_cleaned_df(X_train, X_test, X_train_num_scaled, X_test_num_scaled, cat_to_keep, num_to_keep)
    X_train, y_train = balance_dataset(X_train, y_train)

    best_base_model, _, _ = compare_base_models(models, X_train, y_train, scoring, num_folds)

    param_grid = {
        'bootstrap': [True],
        'max_depth': [80, 90, 100, 110],
        'max_features': [2, 3],
        'min_samples_leaf': [3, 4, 5],
        'min_samples_split': [8, 10, 12],
        'n_estimators': [100, 200, 400, 600]
    }
    best_params = find_best_params(best_base_model[1], param_grid, X_train, y_train)
    #  # {'bootstrap': True, 'max_depth': 80, 'max_features': 2, 'min_samples_leaf': 3, 'min_samples_split': 8, 'n_estimators': 200}
    optimized_model = best_base_model[1](best_params['bootstrap'], best_params['max_depth'], best_params['max_features'],
                       best_params['min_samples_leaf'], best_params['min_samples_split'],
                       best_params['n_estimators'])
    train_evaluate_save(optimized_model, X_train, y_train, X_test, y_test)


def standardize(X_train, X_test):
    X_train_cats = X_train[cat_vars]
    scaler = StandardScaler().fit(X_train[num_vars])
    X_train_num_scaled = scaler.transform(X_train[num_vars])
    X_train_num_scaled = pd.DataFrame(X_train_num_scaled, columns=X_train[num_vars].columns).set_index(X_train.index)

    X_test_num_scaled = scaler.transform(X_test[num_vars])
    X_test_num_scaled = pd.DataFrame(X_test_num_scaled, columns=X_test[num_vars].columns).set_index(X_test.index)

    return X_train_cats, X_train_num_scaled, X_test_num_scaled


def feature_selection(X_train_cats, X_train_num_scaled, y_train):
    print("Starting feature selection")
    num_feat_to_keep = pd.DataFrame(index=X_train_cats.columns)
    cat_feat_to_keep = pd.DataFrame(index=X_train_num_scaled.columns)

    print("Categorical variables analysis")
    chi_analysis(cat_feat_to_keep, X_train_cats, y_train)

    print("Numerical variables analysis")
    anova_analysis(num_feat_to_keep, X_train_num_scaled, y_train)
    tree_analysis(num_feat_to_keep, X_train_num_scaled, y_train)
    lasso_analysis(num_feat_to_keep, X_train_num_scaled, y_train)
    rfe_analysis(num_feat_to_keep, X_train_num_scaled, y_train)

    print("Fetching variables to keep")
    num_feat_to_keep['Discard Nr'] = num_feat_to_keep.apply(lambda x: x.str.findall('Discard').str.len()).sum(
        axis=1).astype(int)
    cat_feat_to_keep['Discard Nr'] = cat_feat_to_keep.apply(lambda x: x.str.findall('Discard').str.len()).sum(
        axis=1).astype(int)

    cat_to_keep = cat_feat_to_keep[cat_feat_to_keep['Discard Nr'] < 1].index.tolist()
    num_to_keep = num_feat_to_keep[num_feat_to_keep['Discard Nr'] < 2].index.to_list()
    return cat_to_keep, num_to_keep


def add_feature_selection_res(df_res, features_to_keep, name):
    df_res[name] = 'Discard'
    for var in features_to_keep:
        df_res.loc[var, name] = 'Keep'


def chi_analysis(num_feat_to_keep, X_train, y_train):
    print("Chi-square analysis")
    chi2_features = SelectKBest(chi2, k=6)
    chi2_features.fit_transform(X_train, y_train)
    features_to_keep = chi2_features.get_feature_names_out()
    add_feature_selection_res(num_feat_to_keep, features_to_keep, 'Chi2')


def anova_analysis(num_feat_to_keep, X_train, y_train):
    print("ANOVA analysis")
    fvalue_selector = SelectKBest(f_classif, k=3)
    fvalue_selector.fit_transform(X_train, y_train)
    features_to_keep = X_train.columns[fvalue_selector.get_support(indices=True)].to_list()
    add_feature_selection_res(num_feat_to_keep, features_to_keep, 'ANOVA')


def tree_analysis(num_feat_to_keep, X_train, y_train):
    print("Tree based analysis")
    clf = ExtraTreesClassifier(n_estimators=50)
    clf = clf.fit(X_train, y_train)
    features_to_keep = pd.Series(clf.feature_importances_, index=X_train.columns).nlargest(3).index
    add_feature_selection_res(num_feat_to_keep, features_to_keep, 'Trees')


def lasso_analysis(num_feat_to_keep, X_train, y_train):
    print("Lasso analysis")
    reg = LassoCV()
    reg.fit(X_train, y_train)
    coef = pd.Series(abs(reg.coef_), index=X_train.columns)  #Check the coefficients associated with each of the variables
    features_to_keep = coef.nlargest(3).index
    add_feature_selection_res(num_feat_to_keep, features_to_keep, 'Lasso Regression')


def rfe_analysis(num_feat_to_keep, X_train, y_train):
    print("Recursive feature extraction analysis")
    rfe = RFE(estimator=RandomForestClassifier(), n_features_to_select=3)
    rfe.fit(X_train, y_train)
    features_to_keep = X_train.columns[rfe.support_]
    add_feature_selection_res(num_feat_to_keep, features_to_keep, 'RFE')


def format_cleaned_df( X_train, X_test, X_train_num_scaled, X_test_num_scaled, cat_to_keep, num_to_keep):
    print("Formating cleaned dataframe")
    X_train = X_train[cat_to_keep].copy()
    X_train[num_to_keep] = X_train_num_scaled[num_to_keep]
    X_test = X_test[cat_to_keep].copy()
    X_test[num_to_keep] = X_test_num_scaled[num_to_keep]
    return X_train, X_test


def balance_dataset(X_train, y_train):
    print("Balacing dataset with SMOTE method")
    print("Before over sampling: ", Counter(y_train))
    over_sample = SMOTE(random_state=7)
    X_train_over, y_train_over = over_sample.fit_resample(X_train, y_train)
    print("After over sampling: ", Counter(y_train_over))
    return X_train_over, y_train_over


def compare_base_models(models, X_train, y_train, scoring, num_folds):
    results = []
    results_means = []
    names = []
    df_res = pd.DataFrame(columns=['model', 'score_accuracy'])
    for name, model in models:
        kfold = KFold(n_splits=num_folds)
        cv_results = cross_val_score(model, X_train, y_train, cv=kfold, scoring=scoring)

        for res in cv_results:
            df_res.loc[len(df_res.index)] = [name, res]

        results.append(cv_results)
        results_means.append(cv_results.mean())
        names.append(name)
        msg = "%s: %f (%f)" % (name, cv_results.mean(), cv_results.std())
        print(msg)

    index_min = max(range(len(results_means)), key=results_means.__getitem__)
    return models[index_min], names, results


def find_best_params(model, param_grid, X_train, y_train):
    grid_search = GridSearchCV(estimator=model, param_grid=param_grid, cv=5, n_jobs=-1, verbose=2)
    grid_result = grid_search.fit(X_train, y_train)
    print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))
    return grid_result.best_params_


def train_evaluate_save(model, X_train, y_train, X_test, y_test):
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)
    accuracy = accuracy_score(y_test, predictions)
    print('Model Performance')
    print('Accuracy: {:0.2f}%'.format(accuracy))
    joblib.dump(model, '../../data/models/popularity.joblib')
    return accuracy


def get_test_train_data(sort):
    filenames = next(walk(BASE_FOLDER), (None, None, []))[2]
    print(filenames)

    train_df, test_df = pd.DataFrame(), pd.DataFrame()
    if len(filenames) == 2:
        train_df = pd.read_csv(filepath_or_buffer=BASE_FOLDER + filenames[0], sep=",")
        test_df = pd.read_csv(filepath_or_buffer=BASE_FOLDER + filenames[1], sep=",")
    elif len(filenames) > 2:
        test_df = pd.read_csv(filepath_or_buffer=BASE_FOLDER + filenames[len(filenames) - 1], sep=",")
        test_df = convert_and_sort_time(test_df, sort)

        train_df = pd.DataFrame()
        for i in range(len(filenames) - 1):
            df_temp = pd.read_csv(filepath_or_buffer=BASE_FOLDER + filenames[i], sep=",")
            df_temp = convert_and_sort_time(df_temp, sort)
            train_df = pd.concat([train_df, pd.DataFrame.from_records(df_temp)])

    return train_df, test_df


def convert_and_sort_time(df, sort):
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df['timestamp'] = [i.replace(tzinfo=datetime.timezone.utc) for i in df['timestamp']]
    if sort:
        df = df.sort_values(by='timestamp', ascending=True)
    return df


def prepare_model_data(df):
    df = df[df['topics_ids'] != -1].copy()
    df['popularity'] = [0 if retweets == 0 else 1 for retweets in df['retweet_count']]
    return df


def split_data(train_df, test_df):
    X_train = train_df.drop('popularity', axis=1)
    y_train = train_df['popularity']
    print(X_train.shape)
    print(y_train.shape)
    X_test = test_df.drop('popularity', axis=1)
    y_test = test_df['popularity']
    print(X_test.shape)
    print(y_test.shape)
    return X_train, y_train, X_test, y_test