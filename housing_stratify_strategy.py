# By:   K. Ng
# Date: 2019/05
#
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedShuffleSplit

#
#   Data Splitting needs to be of similar statistical distributions (i.e. Stratified)
#   More than one might be needed: create various strategies here as fucntions


def strat_income_cat(housing):
    """
    From various income levels, create income categories 1-5

    :param housing: Pandas DataFrame
    :return: Pandas DataFrams strat_train_set, strat_test_set

    """

    housing['income_cat'] = np.ceil(housing['median_income']/1.5)
    housing['income_cat'].where(housing['income_cat']<5, 5.0, inplace=True)
    housing['income_cat'].where(housing['income_cat']<5, other=5.0, inplace=True)

    strat_split = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)

    for train_index, test_index in strat_split.split(housing, housing['income_cat']):
        strat_train_set = housing.loc[train_index]
        strat_test_set = housing.loc[test_index]

    dist_pcts = pd.concat([
        housing['income_cat'].value_counts() / len(housing) * 100,
        strat_train_set['income_cat'].value_counts() / len(strat_train_set) * 100,
        strat_test_set['income_cat'].value_counts() / len(strat_test_set) * 100
    ], axis=1)

    dist_pcts.columns = ['%in_cat full', '%in_cat train', '%in_cat test']
    dist_pcts['train var'] = abs(1 - (dist_pcts['%in_cat train']/dist_pcts['%in_cat full']))
    dist_pcts['test var'] = abs(1 - (dist_pcts['%in_cat test']/ dist_pcts['%in_cat full']))

    print('\nStratification Check:')
    print('\tTraining set: max dev of %_distribution:\t{:2.2%}'.format(dist_pcts['train var'].max()))
    print('\tTest set: max dev of %_distribution:\t\t{:2.2%}\n'.format(dist_pcts['test var'].max()))

    print('\tTraining set: size:\t{}'.format(len(strat_train_set)))
    print('\tTest set: size:\t\t{}'.format(len(strat_test_set)))

    return strat_train_set, strat_test_set
