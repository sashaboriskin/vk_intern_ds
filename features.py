import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import haversine_distances
from sklearn.model_selection import train_test_split

def calculate_haversine_distances(base_coords, compare_coords):
    distances = haversine_distances(base_coords, compare_coords) * 6371000/1000  # in km
    return distances

def generate_features():
    train_df = pd.read_csv('data/train.csv')
    test_df = pd.read_csv('data/test.csv')
    features_df = pd.read_csv('data/features.csv')

    train_coords = train_df[['lat', 'lon']].apply(np.radians)
    test_coords = test_df[['lat', 'lon']].apply(np.radians)
    features_coords = features_df[['lat', 'lon']].apply(np.radians)

    train_distances = calculate_haversine_distances(train_coords, features_coords)
    test_distances = calculate_haversine_distances(test_coords, features_coords)

    train_closest_feature_indices = np.argmin(train_distances, axis=1)
    test_closest_feature_indices = np.argmin(test_distances, axis=1)

    train_features = features_df.iloc[train_closest_feature_indices].reset_index(drop=True)
    test_features = features_df.iloc[test_closest_feature_indices].reset_index(drop=True)

    train_df_final = train_df.drop(columns=['lat', 'lon'])
    test_df_final = test_df.drop(columns=['lat', 'lon'])

    train_features_final = train_features.drop(columns=['lat', 'lon'])
    test_features_final = test_features.drop(columns=['lat', 'lon'])

    X_train = pd.concat([train_df_final.drop(columns=['score']), train_features_final], axis=1)
    X_train['random'] = np.random.rand() # random feature for feature selection
    y_train = train_df_final['score']
    X_test = pd.concat([test_df_final, test_features_final], axis=1)
    X_test['random'] = np.random.rand()

    X_train, X_eval, y_train, y_eval = train_test_split(X_train, y_train, test_size=0.2, random_state=42)
    return X_train, X_eval, y_train, y_eval, X_test
