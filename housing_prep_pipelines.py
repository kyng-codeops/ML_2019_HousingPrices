# By:   K. Ng
# Date: 2019/05
#
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin

#
# sklearn boiler-plate code:
# Object that converts Pandas DFs into numbers ONLY
#


class DataFrameSelector(BaseEstimator, TransformerMixin):
    def __init__(self, attribute_names):
        self.attribute_names = attribute_names

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        return X[self.attribute_names].values

#
# sklearn boiler-plate code:
# for parametric feature selection (i.e. Feature Engineering)
#


class CombinedAttributesAdder(BaseEstimator, TransformerMixin):

    rooms_ix, bedrooms_ix, population_ix, household_ix, income_ix = 3, 4, 5, 6, 7

    def __init__(self, add_bedrooms_per_room=False, add_income_ratios=False):  # no *args or **kargs
        self.add_bedrooms_per_room = add_bedrooms_per_room
        self.add_income_ratios = add_income_ratios

    def fit(self, X, y=None):
        return self  # nothing else to do

    def transform(self, X, y=None):
        rooms_per_household = X[:, self.rooms_ix] / X[:, self.household_ix]
        population_per_household = X[:, self.population_ix] / X[:, self.household_ix]

        if self.add_bedrooms_per_room:
            bedrooms_per_room = X[:, self.bedrooms_ix] / X[:, self.rooms_ix]
            return np.c_[X, rooms_per_household, population_per_household,
                         bedrooms_per_room
            ]
        elif self.add_income_ratios:
            bedrooms_per_room = X[:, self.bedrooms_ix] / X[:, self.rooms_ix]
            income_to_bedrooms = X[:, self.income_ix] / X[:, self.bedrooms_ix]
            income_to_rooms = X[:, self.income_ix] / X[:, self.rooms_ix]
            return np.c_[X, rooms_per_household, population_per_household,
                         bedrooms_per_room,
                         income_to_bedrooms, income_to_rooms
            ]
        else:
            return np.c_[X, rooms_per_household, population_per_household]

    def extras(self):
        if self.add_bedrooms_per_room:
            return ['rooms_per_hhold', 'pop_per_hhold', 'bedrooms_per_rm']
        elif self.add_income_ratios:
            return ['rooms_per_hhold', 'pop_per_hhold', 'bedrooms_per_rm', 'income_to_bedrm', 'income_to_rm']
        else:
            return ['rooms_per_hhold', 'pop_per_hhold']

#
#   K. Ng create
#   Selectable preprocessing pipelines with various feature-sets
#
from sklearn.impute import SimpleImputer as Imputer
from sklearn.preprocessing import OneHotEncoder
# from sklearn.preprocessing import OrdinalEncoder      # too basic
# from sklearn.preprocessing import CategoricalEncoder  # deprecated
from sklearn.preprocessing import StandardScaler

from sklearn.pipeline import Pipeline
from sklearn.pipeline import FeatureUnion


class basic:
    cat_attribs = ['ocean_proximity']
    rem_attribs = cat_attribs + ['median_house_value']

    def __init__(self, housing, strat_train_set, strat_test_set):
        self.housing = housing
        self.strat_train_set = strat_train_set
        self.strat_test_set = strat_test_set

        self.attr_adder = CombinedAttributesAdder(add_bedrooms_per_room=False)
        self.headings = []


    def makeit(self):
        self.housing_num = self.housing.drop(self.rem_attribs, axis=1)
        self.num_attribs = list(self.housing_num)

        self.num_pipeline = Pipeline([
            ('selector', DataFrameSelector(self.num_attribs)),
            ('imputer', Imputer(strategy="median")),
            ('attribs_adder', self.attr_adder),
            ('std_scaler', StandardScaler()),
        ])

        self.cat_pipeline = Pipeline([
            ('selector', DataFrameSelector(self.cat_attribs)),
            ('cat_encoder', OneHotEncoder(sparse=False))
        ])

        self.full_pipeline = FeatureUnion(transformer_list=[
            ('num_pipeline', self.num_pipeline),
            ('cat_pipeline', self.cat_pipeline),
        ])

        self.train_labels = self.strat_train_set['median_house_value'].copy().to_numpy()
        self.strat_train_set.drop('median_house_value', axis=1)
        self.train_features_prepared = self.full_pipeline.fit_transform(self.strat_train_set)

        self.test_lables = self.strat_test_set['median_house_value'].to_numpy()
        self.strat_test_set.drop('median_house_value', axis=1)
        self.test_features_prepared = self.full_pipeline.fit_transform(self.strat_test_set)

        self.cat_encoder = self.cat_pipeline.named_steps['cat_encoder']
        self.cat_onehot_attribs = list(self.cat_encoder.categories_[0])
        self.headings = self.num_attribs  + self.attr_adder.extras() + self.cat_onehot_attribs

        print('\n' + '='*80)
        # print('\nPipeline: {}'.format(self.attr_adder))
        print('Pipeline training array.shape:\t', self.train_features_prepared.shape)
        print('Pipeline test array.shape:\t\t', self.test_features_prepared.shape)

        return self.train_labels, self.train_features_prepared, self.test_lables, self.test_features_prepared


    def header(self):
        return self.headings


class with_bedrms(basic):

    def __init__(self, housing, strat_train_set, strat_test_set):
        self.housing = housing
        self.strat_train_set = strat_train_set
        self.strat_test_set = strat_test_set

        self.attr_adder = CombinedAttributesAdder(add_bedrooms_per_room=True)
        self.headings = []

    pass


class with_income_ratios(basic):

    def __init__(self, housing, strat_train_set, strat_test_set):
        self.housing = housing
        self.strat_train_set = strat_train_set
        self.strat_test_set = strat_test_set

        self.attr_adder = CombinedAttributesAdder(add_income_ratios=True)
        self.headings = []

    pass







