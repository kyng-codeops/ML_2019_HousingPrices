import numpy as np
#
#   Model Testing
#
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import cross_val_score
# from sklearn.externals import joblib
import joblib


def display_scores(target_hi, target_lo, cv_val, rmse_orig, cv_scores, title):
    """
    Simple cli to rapid print model scoring summaries

    :param target_hi:   Max actual median_house_values
    :param target_lo:   Min actual median_house_values
    :param rmse_orig:   rmse of the full training dataset
    :param cv_scores:   Array of rmse values from cross_val sub-set training
    :param title:       Text string to describe the model
    :return:
    """

    print('\n' + '-'*80 + '\n')
    print('{}'.format(title))
    print('\tK-Folds:\t{}'.format(cv_val))
    print('\tFull_train_set Mean Error: +/- ${:.2f}'.format(rmse_orig.mean()))
    print('\tFold_sets Mean Error: +/- ${:.2f}'.format(cv_scores.mean()))
    print('\tCross-Validation Chg in Errors: {:2.2%}'.format((rmse_orig.mean()-cv_scores.mean())/cv_scores.mean()))
    print('\tCross-Validation Std Dev of Errors: {:.2f}'.format(cv_scores.std()))
    # print('\tRMS_Err vs Target High: {:2.2%}'.format((target_hi-rmse_orig.mean())/target_hi))
    # print('\tRMS_Err vs Target Low: {:2.2%}'.format(abs(target_lo - rmse_orig.mean()) / target_lo))

    return


def rmse_test(model, cv_val, train_labels, train_features_prepared, test_lables, test_features_prepared):
    """
    Root Mean Sq Errors is just one way to score ML models

    :param model: a sklean regressor model 'LinearRegression()', 'DecisionTreeRegressor()', or 'RandomForestRegressor()'
    :param cv_val: integer K-folds or CV number
    :param train_labels: numpy array
    :param train_features_prepared: numpy array
    :param test_lables: numpy array
    :param test_features_prepared: numpy array
    :return:
    """

    title = str(model)
    model.fit(train_features_prepared, train_labels)
    test_predictions = model.predict(test_features_prepared)

    model_mse = mean_squared_error(test_lables, test_predictions)
    model_rmse = np.sqrt(model_mse)

    scores = cross_val_score(model, train_features_prepared, train_labels, scoring='neg_mean_squared_error', cv=cv_val)
    lin_rmse_scores = np.sqrt(-scores)
    display_scores(test_lables.max(), test_lables.min(), cv_val, model_rmse, lin_rmse_scores, title)

    return
