# By:   K. Ng
# Date: 2019/05
#
import housing_stratify_strategy
import housing_prep_pipelines as housing_prep
import housing_model_scoring

import os
import tarfile
# from six.moves import urllib
import urllib
import pandas as pd
from sklearn.externals import joblib

#
# Program Constants
#
DOWNLOAD_ROOT = "https://raw.githubusercontent.com/ageron/handson-ml/master/"
HOUSING_PATH = os.path.join("datasets", "housing")
HOUSING_URL = DOWNLOAD_ROOT + "datasets/housing/housing.tgz"


#
# Program Functions
#
def fetch_housing_data(housing_url=HOUSING_URL, housing_path=HOUSING_PATH):
    if not os.path.isdir(housing_path):
        os.makedirs(housing_path)
    tgz_path = os.path.join(housing_path, "housing.tgz")
    urllib.request.urlretrieve(housing_url, tgz_path)   #nosec
    housing_tgz = tarfile.open(tgz_path)
    housing_tgz.extractall(path=housing_path)
    housing_tgz.close()


def load_housing_data(housing_path=HOUSING_PATH):
    csv_path = os.path.join("~/", "PycharmProjects", "ML_2019_HousingPrices", housing_path, "housing.csv")
    return pd.read_csv(csv_path)


def save_models_as_files(my_model):
    # TODO: placeholder, need to pass key,val of model and filenames to save
    joblib.dump(my_model, "six_moves_model.pkl")


def load_saved_models_frm_files():
    # TODO: placeholder, need to pass key,val of model and filenames to load
    my_model = joblib.load("six_moves_model.pkl")


def main():
    """
    Main Program to train and evaluate model inference performance

    :return: None
    """

    housing = load_housing_data()

    # Only 1 stratification strategy so no list of various strategies needed
    strat_train, strat_test = housing_stratify_strategy.strat_income_cat(housing)

    # Multiple preprocessing pipelines to study
    pipe_obj_1 = housing_prep.basic(housing, strat_train, strat_test)
    pipe_obj_2 = housing_prep.with_bedrms(housing, strat_train, strat_test)
    pipe_obj_3 = housing_prep.with_income_ratios(housing, strat_train, strat_test)

    prep_list = [
        pipe_obj_1,
        pipe_obj_2,
        pipe_obj_3
    ]

    # Multiple models of interest
    from sklearn.linear_model import LinearRegression
    from sklearn.tree import DecisionTreeRegressor
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.svm import SVR

    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import PolynomialFeatures

    PolyRegression = Pipeline([
        ('polynomial_features', PolynomialFeatures(degree=4, include_bias=False)),
        ('linear_regression', LinearRegression())
    ])

    ml_list = [
        LinearRegression(),
        DecisionTreeRegressor(),
        RandomForestRegressor(n_estimators=10),
        SVR(gamma='scale')
    ]

    k_folds = 3

    # Comparisons #1
    #
    print('\n\n\n' + '*' * 80 + '\n')
    print('\n*** Models vs Models ***\n')

    for i in range(0, len(prep_list)):
        prep_version = prep_list[i]
        train_labels, train_feats_preproc, test_lables, test_feats_preproc = prep_version.makeit()

        print('pipeline prep #{}: {}'.format((i+1), prep_version.header()))

        for model in ml_list:
            housing_model_scoring.rmse_test(model, k_folds, train_labels, train_feats_preproc, test_lables, test_feats_preproc)

    # Comparisons #2
    #
    print('\n\n\n' + '*' * 80 + '\n')
    print('\n*** Pipelines vs Pipelines ***\n')

    for model in ml_list:
        for prep_version in prep_list:
            train_labels, train_feats_preproc, test_lables, test_feats_preproc = prep_version.makeit()
            housing_model_scoring.rmse_test(model, k_folds, train_labels, train_feats_preproc, test_lables, test_feats_preproc)

    # How about tweaking the RandomForest hyper-parameters?
    #
    from sklearn.model_selection import GridSearchCV

    print('\n\n\n' + '*' * 80 + '\n')
    print('\n*** RandomForest HyperParam GridSearch ***\n')
    model = RandomForestRegressor()
    param_grid = [
        {
            'bootstrap': [True, False],
            'n_estimators': [3, 10, 30, 90],
            'max_features': [2, 3, 4, 6, 8],
        }
    ]

    for prep_version in prep_list:
        train_labels, train_feats_preproc, test_lables, test_feats_preproc = prep_version.makeit()

        grid_search = GridSearchCV(model, param_grid, cv=k_folds, scoring='neg_mean_squared_error')
        grid_search.fit(train_feats_preproc, train_labels)
        print('\nBest GSCV RandomForest params: {}'.format(grid_search.best_params_))

        housing_model_scoring.rmse_test(
            grid_search.best_estimator_,
            k_folds,
            train_labels,
            train_feats_preproc,
            test_lables,
            test_feats_preproc
        )

        feature_ranks = grid_search.best_estimator_.feature_importances_
        for f in sorted(zip(feature_ranks, prep_version.header()), reverse=True):
            print('{:2.2}%\t\t{}'.format(f[0], f[1]))


if __name__ == '__main__':
    main()
