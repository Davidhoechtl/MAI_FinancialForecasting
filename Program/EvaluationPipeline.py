import pandas as pd

from Forecasting.ARMA import ARMAForecastingModel
from Forecasting.EvaluationModelBase import ForecastingModelBase
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.metrics import mean_squared_error, root_mean_squared_error
import numpy as np
import pandas as pd

from Forecasting.GRU import GRUForecastingModel
from Forecasting.LSTM import LSTMForecastingModel


def evaluate_model_on_classification(
        model: ForecastingModelBase,
        feature_matrix: pd.DataFrame,
        predictor_cols: list[str],
        target_col: str,
        target_horizon_in_days: int):
    """
    Evaluates a classification model using expanding window cross-validation.
    Args:
        model: Machine Learning modlel implementing ForecastingModelBase.
        feature_matrix: feature matrix DataFrame (result of Feature Matrix Pipeline).
        predictor_cols: features that should be used for prediction.
        target_col: target column name.
        target_horizon_in_days: time horizon of the target variable in days

    Returns: Dictionary with averaged evaluation metrics (accuracy, precision, recall, f1_score).
    """

    # Splits generation
    splits = train_test_split_expanding_window(
        feature_matrix,
        predictor_cols,
        target_col,
        initial_train_size=200,
        test_size=100,
        padding=target_horizon_in_days
    )

    if not splits:
        print("No splits generated. Check data size.")
        return None

    print("Evaluating Model: ", model.name)
    print_split_info(splits)

    scores = []

    for split in splits:
        train_X, train_y, gap_X, gap_y, test_X, test_y = split
        model.train(train_X, train_y)
        predictions = model.predict(test_X, gap_X)

        # Calculate metrics using sklearn (handles zero division automatically)
        accuracy = accuracy_score(test_y, predictions)
        precision = precision_score(test_y, predictions, zero_division=0)
        recall = recall_score(test_y, predictions, zero_division=0)
        f1 = f1_score(test_y, predictions, zero_division=0)

        print(f"Split Accuracy: {accuracy:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}, F1-Score: {f1:.4f}")

        scores.append({
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1
        })

    # Aggregation
    avg_scores = pd.DataFrame(scores).mean()

    print(f"Model: {model.name}")
    print(f"Final Accuracy: {avg_scores['accuracy']:.4f}")
    print(f"Final Precision: {avg_scores['precision']:.4f}")
    print(f"Final Recall: {avg_scores['recall']:.4f}")
    print(f"Final F1-Score: {avg_scores['f1_score']:.4f}")

    return avg_scores.to_dict()

def evaluate_model_on_regression(
        model: ForecastingModelBase,
        feature_matrix: pd.DataFrame,
        predictor_cols: list[str],
        target_col: str,
        target_horizon_in_days: int):
    """
    Evaluates a regression model using expanding window cross-validation.
    Args:
        model: Machine Learning modlel implementing ForecastingModelBase.
        feature_matrix: feature matrix DataFrame (result of Feature Matrix Pipeline).
        predictor_cols: features that should be used for prediction.
        target_col: target column name.
        target_horizon_in_days: time horizon of the target variable in days

    Returns: Dictionary with averaged evaluation metrics (MSE, RMSE).
    """
    # Splits generation
    splits = train_test_split_expanding_window(
        feature_matrix,
        predictor_cols,
        target_col,
        initial_train_size=200,
        test_size=100,
        padding=target_horizon_in_days
    )

    # Safety check for empty splits
    if not splits:
        print("No splits generated. Check data size.")
        return None

    print("Evaluating Model: ", model.name)
    print_split_info(splits)

    scores = []

    for split in splits:
        train_X, train_y, gap_X, gap_y, test_X, test_y = split
        model.train(train_X, train_y)

        if isinstance(model, ARMAForecastingModel) is True:
            # Because of the nature of auto regressive models, we need to provide the gap and ground truth data to the prediction function
            predictions = model.predict_arma(test_X, test_y, gap_X, gap_y)
        else:
            predictions = model.predict(test_X, gap_X)

        # assert len(predictions) == len(test_y) - 7
        # assert len(predictions) == len(test_y), "Prediction and ground truth lengths do not match."

        # Calculate Metrics
        if (isinstance(model, LSTMForecastingModel) or
            isinstance(model, GRUForecastingModel)) is True:
            # LSTM model needs to skip first 7 predictions due to sequence length
            # Todo: Refactor LSTM to avoid this hack
            mse = mean_squared_error(test_y[7:], predictions)
            rmse = np.sqrt(mse)
        else:
            mse = mean_squared_error(test_y, predictions)
            rmse = np.sqrt(mse)

        print(f"Split MSE: {mse:.4f}, RMSE: {rmse:.4f}")
        scores.append({'MSE': mse, 'RMSE': rmse})

    # Aggregation
    avg_scores = pd.DataFrame(scores).mean()

    print(f"Model: {model.name}")
    print(f"Final MSE: {avg_scores['MSE']:.4f}")
    print(f"Final RMSE: {avg_scores['RMSE']:.4f}")

    return avg_scores.to_dict()

def train_test_split_expanding_window(
        feature_matrix: pd.DataFrame,
        predictor_cols: list[str],
        target_col: str,
        initial_train_size: int,
        test_size: int,
        padding: int = 0 ) -> list[tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.Series, pd.DataFrame, pd.Series]]:
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
    start_test = initial_train_size

    while start_test + test_size <= total_size:
        end_of_train = start_test - padding
        train_data = feature_matrix.iloc[:end_of_train]
        test_data = feature_matrix.iloc[start_test:start_test + test_size]

        gap_x = feature_matrix.iloc[end_of_train:start_test][predictor_cols]
        gap_y = feature_matrix.iloc[end_of_train:start_test][target_col]

        train_X = train_data[predictor_cols]
        train_y = train_data[target_col]
        test_X = test_data[predictor_cols]
        test_y = test_data[target_col]

        splits.append((train_X, train_y, gap_x, gap_y, test_X, test_y))
        start_test += test_size  # Move the window forward

    return splits

def print_split_info(splits: list[tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.Series, pd.DataFrame, pd.Series]]):
    print("Number of splits generated: ", len(splits))
    for i, split in enumerate(splits):
        train_X, train_y, gap_X, gap_y, test_X, test_y = split
        print(f"Split {i+1}:")
        print(f"  Train size: {len(train_X)}")
        print(f"  Gap size: {len(gap_X)}")
        print(f"  Test size: {len(test_X)}")