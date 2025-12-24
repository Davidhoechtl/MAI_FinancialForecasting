import pandas as pd
from Forecasting.EvaluationModelBase import ForecastingModelBase

def evaluate_model_on_classification(model : ForecastingModelBase, feature_matrix: pd.DataFrame, predictor_cols: list[str], target_col: str):
    # Todo: implement classification evaluation here
    # return should be confusion matrix: accuracy, precision, recall, f1-score
    return None

def evaluate_model_on_regression(model : ForecastingModelBase, feature_matrix: pd.DataFrame, predictor_cols: list[str], target_col: str):
    #Todo: implement regression evaluation here
    # return should be: MSE, RMSE
    return None

def train_test_split_expanding_window(
        feature_matrix: pd.DataFrame,
        predictor_cols: list[str],
        target_col: str,
        initial_train_size: int,
        test_size: int,
        padding: int = 0 ) -> list[tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.Series]]:
    """
    Splits the data into training and testing sets using an expanding window approach.

    Parameters:
        feature_matrix (pd.DataFrame): The complete feature matrix.
        predictor_cols (list[str]): List of predictor column names.
        target_col (str): The target column name.
        initial_train_size (int): The initial size of the training set.
        test_size (int): The size of the test set for each iteration.
        padding (int): Number of rows to skip between training and testing sets (default is 0).

    Returns:
        list of tuples: Each tuple contains (train_X, train_y, test_X, test_y) for each split.
    """
    splits = []
    total_size = len(feature_matrix)
    start_test = initial_train_size + padding

    while start_test + test_size <= total_size:
        train_data = feature_matrix.iloc[:start_test]
        test_data = feature_matrix.iloc[start_test:start_test + test_size]

        train_X = train_data[predictor_cols]
        train_y = train_data[target_col]
        test_X = test_data[predictor_cols]
        test_y = test_data[target_col]

        splits.append((train_X, train_y, test_X, test_y))
        start_test += test_size  # Move the window forward

    return splits