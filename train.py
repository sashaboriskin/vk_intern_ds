from features import generate_features

import pandas as pd
import numpy as np
from sklearn.metrics import mean_absolute_error
from catboost import CatBoostRegressor

def fit():
    X_train, X_eval, y_train, y_eval, X_test = generate_features()

    feature_selection_model = CatBoostRegressor(verbose=False)
    feature_selection_model.fit(X_train, y_train)
    preds = feature_selection_model.predict(X_eval)

    feature_importance = feature_selection_model.feature_importances_
    sorted_idx = np.argsort(feature_importance)

    f_i = pd.DataFrame({'feature_importance' : feature_importance[sorted_idx],
                        'column': np.array(X_eval.columns)[sorted_idx]})
    rand = f_i[f_i['column'] == 'random']['feature_importance'].values.item()
    columns_to_drop = list(f_i[f_i['feature_importance'] <= rand].column.values)
    X_train = X_train.drop(columns=columns_to_drop)
    X_test = X_test.drop(columns=columns_to_drop)

    model = CatBoostRegressor(iterations=1000,
                              loss_function='MAE',
                              verbose=False)
    model.fit(X_train, y_train)
    preds = model.predict(X_eval)
    print('eval mae: ', mean_absolute_error(y_eval, preds))
    preds_test = model.predict(X_test)
    return preds_test
