import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import pingouin as pg
from sklearn.model_selection import RepeatedKFold, cross_validate
from sklearn.linear_model import LinearRegression

#*** These analyses are based on the code provided by Cooper and Ritchey (2022), which can be found here: https://github.com/memobc/paper-vividness-features
#*** They have been extended to allow for an additional manipulation of an age group (both young and older adults have been tested) and an additional dependent variable - a confidence rating¶

def quality_check(my_data):
    # uses RTs and performance to check quality of data and
    # mark subjects to exclude

    # set up an empty list to store subject IDs to exclude
    exclude_subs = []
    Nsubs = len(my_data.participant.unique())

    # a) RT # --------------------- #
    # calculate median RT across all types of memory test response:
    rt_cols = ['resp_ret_vividness.rt', 'resp_ret_confidence.rt',
               'resp_gistmem1.rt', 'resp_gistmem2.rt', 'resp_gistmem3.rt',
               'resp_detailmem1.rt', 'resp_detailmem2.rt', 'resp_detailmem3.rt']
    rt_data = my_data[['participant'] + rt_cols].melt(id_vars=['participant']).groupby(
        ['participant']).median(numeric_only=True).reset_index()

    # find any subjects <= .5s and add to exclude list (cannot respond for .25 before clock starts)
    ps = rt_data.loc[rt_data['value'] <= .5, 'participant'].to_list()
    exclude_subs.extend(ps)
    print('\nNumber of subjects with median RT <= .75s --', len(ps), 'out of', Nsubs, 'subjects')

    # b) Same Key # --------------------- #
    # calculate the frequency with which each key was used
    key_cols = ['resp_gistmem1.keys', 'resp_gistmem2.keys', 'resp_gistmem3.keys',
                'resp_detailmem1.keys', 'resp_detailmem2.keys', 'resp_detailmem3.keys']
    key_data = my_data[['participant'] + key_cols].melt(id_vars=['participant'])
    key_data = key_data.groupby(['participant'])['value'].value_counts().unstack().reset_index()

    # now work out if any % of presses is > 75%
    key_cols = [1.0, 2.0, 3.0, 4.0]
    # 144 because 24 trials * 6 questions (vividness and confidence were tested using a continuous scale)
    key_data[key_cols] = (key_data[key_cols] / 144) * 100
    key_data = key_data[key_cols].max(axis=1).to_frame()
    key_data['participant'] = my_data['participant'].unique()

    ps = key_data.loc[key_data[0] > 75, 'participant'].to_list()
    exclude_subs.extend(ps)
    print('\nNumber of subjects with consistent key presses (> 75% same key) --', len(ps), 'out of', Nsubs, 'subjects')

    # c) Gist Memory # --------------------- #
    gist = my_data[['participant',
                    'resp_gistmem1.corr', 'resp_gistmem2.corr', 'resp_gistmem3.corr']].melt(id_vars='participant')
    gist = gist.groupby(['participant']).mean(numeric_only=True).reset_index()

    # exclude? chance is 25%
    ps = gist.loc[gist['value'] <= .3, 'participant'].to_list()
    exclude_subs.extend(ps)
    print('\nNumber of subjects with gist memory <= 30% --', len(ps), 'out of', Nsubs, 'subjects')

    # ***return full list of excluded subjects***
    exclude_subs = np.unique(exclude_subs).tolist()  # removing duplicates from above exclusions

    # remove from my_data
    if len(exclude_subs) > 0:
        message = '\nRemoving subjects from data .... ' + ','.join(map(str,exclude_subs))
        print(message)
        # remove from df
        my_data = my_data[~my_data['participant'].isin(exclude_subs)]

    return my_data, exclude_subs
# -------------------------------------- #


def format_memory_data(my_data):
    # formats the data from questions into responses per feature:

    # rename columns for merging across questions:
    feature1 = my_data[['participant', 'event_id', 'q1_type', 'resp_gistmem1.corr', 'resp_detailmem1.corr']]
    feature1.columns = ["participant", "event", "type", "gist", "detail"]

    feature2 = my_data[['participant', 'event_id', 'q2_type', 'resp_gistmem2.corr', 'resp_detailmem2.corr']]
    feature2.columns = ["participant", "event", "type", "gist", "detail"]

    feature3 = my_data[['participant', 'event_id', 'q3_type', 'resp_gistmem3.corr', 'resp_detailmem3.corr']]
    feature3.columns = ["participant", "event", "type", "gist", "detail"]

    # concatenate into long format
    feature_data = pd.concat([feature1, feature2, feature3], axis=0, ignore_index=True)

    # now convert to wide format and merge with vividness
    feature_wide = feature_data.pivot_table(index=['participant', 'event'], columns='type').reset_index()

    # add vividness and confidence ratings, by event, to feature_data
    memory_data = my_data[['participant', 'event_id', 'resp_ret_vividness.keys', 'resp_ret_confidence.keys', 'age']]
    memory_data.columns = ["participant", "event", "vividness", "confidence", "age"]
    memory_data = memory_data.merge(feature_wide, on=['participant', 'event'])

    # format column names so not tuples
    viv_columns = memory_data.columns[5:11]
    memory_data.columns = memory_data.columns[0:5].tolist() + ['_'.join(i) for i in viv_columns]

    return memory_data
# -------------------------------------- #


def ratings_correlations(data, features, rating_type):
    # runs within-subject correlations between vividness
    # and (continuous) memory attributes
    # returns spearman r
    columns = features + ["age"]
    cors = pd.DataFrame(index=data['participant'].unique(),
                        columns=columns)
    for p in data['participant'].unique():
        sub_data = data[data['participant'] == p]

        # correlate with vividness
        sub_cors = sub_data[[rating_type] + features].corr(method="spearman").loc[rating_type,features].astype('float')
        cors_idx = (np.isnan(sub_cors)) | (np.round(np.abs(sub_cors),1) == 1)
        sub_cors[cors_idx] = 0
        cors.loc[p,features] = sub_cors
        cors.loc[p, 'age'] = sub_data['age'].unique()[0]

    return cors.reset_index()

def regressions(X, y, plottitle):
    kfold = RepeatedKFold(n_splits=10, n_repeats=5, random_state=1)

    included_features = X.columns.tolist()
    X_orig = X
    X_orig = X_orig.astype('float')
    model_weights = []
    model_scores = []
    for m in range(len(included_features)):

        X = X_orig[included_features]

        # run cross-validation (note. LR() adds intercept by default):
        models = cross_validate(LinearRegression(), X, y, cv=kfold, scoring='r2', return_estimator=True)

        # first, let's store the r-squared scores, and show the average:
        scores = models['test_score']
        model_scores.append(scores)
        print('Features =', included_features)
        print('R-squared =', np.round(np.mean(scores), 5), '( +/-', np.round(np.std(scores), 5), ')\n')

        # find the weights and drop the lowest feature
        features = []
        for idx, model in enumerate(models['estimator']):
            features.append(pd.DataFrame(model.coef_,
                                         index=included_features, columns=['model_' + str(idx)]))
        weights = pd.concat(features, axis=1, ignore_index=True).transpose()
        model_weights.append(weights)  # beta values

        lowest_feature = weights.mean().sort_values().index[0]
        included_features.remove(lowest_feature)


    print('\n FEATURE ACCOUNTING FOR MOST VARIANCE: \n\n' + str(X.columns.tolist()))

    print('\n*BEST MODEL*:\n')

    rsquared_values = pd.DataFrame(model_scores).mean(axis=1)
    best_model = np.where(rsquared_values == rsquared_values.max())[0].tolist()[0]

    print('R-squared =', np.round(rsquared_values[best_model], 4))
    print('Features =', model_weights[best_model].columns.to_list(), '\n')

    return model_weights[0]

def content_specificity_t_test(cor_params, rating):
    tests = []
    for c in cor_params.content.unique():
        for s in cor_params.specificity.unique():
            t = pg.ttest(cor_params.loc[
                             (cor_params.content == c) & (cor_params.specificity == s), rating],
                         0).round(4)
            t['content'] = c
            t['specificity'] = s
            tests.append(t)

    tests = pd.concat(tests)

    # add bonferroni correction:
    tests["p-bonf"] = pg.multicomp(tests["p-val"].tolist(), alpha=.05, method="bonf")[1]
    return tests