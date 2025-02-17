import numpy as np
from sklearn.model_selection import GridSearchCV
from skopt import BayesSearchCV
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.base import BaseEstimator
from sklearn.metrics import make_scorer, accuracy_score, precision_score, recall_score, f1_score
from typing import Dict, Union
from sklearn.metrics._scorer import _Scorer


class ModelTuning:
    """
    A class for tuning hyperparameters of machine learning models using Bayesian optimization.

    Attributes:
        model_name (str): The name of the model to tune (either 'knn' or 'random_forest').
        model (BaseEstimator): The model object initialized based on the selected model.
        param_grid (dict): The hyperparameter grid for the selected model.
    """

    def __init__(self, model_name: str) -> None:
        """
        Initializes the ModelTuning object with the selected model and its hyperparameter grid.

        Args:
            model_name (str): The name of the model to use ('knn' or 'random_forest').

        Raises:
            ValueError: If an unsupported model name is provided.
        """
        self.model_name = model_name
        self.model = self._select_model()  # Initialize the model based on the model_name.
        self.param_grid = self._select_hyperparameters()  # Set the hyperparameter grid for the model.

    def _select_model(self) -> BaseEstimator:
        """
        Selects and returns the appropriate model based on the model name.

        Returns:
            BaseEstimator: A machine learning model (either KNeighborsClassifier or RandomForestClassifier).

        Raises:
            ValueError: If the model name is unsupported.
        """
        if self.model_name == "knn":
            model = KNeighborsClassifier(n_jobs=1)  # Initialize KNeighborsClassifier with parallelism for neighbors.
        elif self.model_name == "random_forest":
            model = RandomForestClassifier(n_jobs=-1)  # Initialize RandomForestClassifier with all cores for training.
        else:
            raise ValueError("Unsupported model type.")  # Raise an error if the model is not supported.
        
        return model

    def _select_hyperparameters(self) -> Dict[str, Union[str, int]]:
        """
        Selects the hyperparameter grid for the specified model.

        Returns:
            dict: A dictionary of hyperparameters and their respective search space.

        Raises:
            ValueError: If the model type is unsupported.
        """
        if self.model_name == "knn":
            param_grid = {
                'n_neighbors': list(range(3, 51)),  # Number of neighbors to consider in KNN (from 3 to 50).
                'metric': ['euclidean', 'manhattan', 'cosine'],  # Distance metrics for KNN.
                'weights': ['uniform', 'distance']  # Weighting function for KNN (uniform or distance-based).
            }
        elif self.model_name == "random_forest":
            param_grid = {
                'n_estimators': [50, 100, 200],  # Number of trees in the forest.
                'max_depth': [None, 10, 20, 30]  # Maximum depth of the trees in the forest.
            }
        else:
            raise ValueError("Unsupported model type.")  # Raise an error if the model is not supported.
        
        return param_grid

    def _select_scoring(self) -> Dict[str, _Scorer]:
        """
        Selects the scoring metrics to evaluate the model.

        Returns:
            dict: A dictionary of scoring metrics (accuracy, precision, recall, f1_score).
        """
        scoring = {
            "accuracy": make_scorer(accuracy_score),  # Accuracy score for model performance.
            "precision": make_scorer(precision_score, average="macro"),  # Precision score for model performance.
            "recall": make_scorer(recall_score, average="macro"),  # Recall score for model performance.
            "f1_score": make_scorer(f1_score, average="macro")  # F1 score for model performance.
        }

        return scoring

    def tune_hyperparameters(self, X: np.ndarray, y: np.ndarray, metric_optmize: str = "f1_score", n_iter: int = 20, cv: int = 5, random_state: int = 42):
        """
        Tunes the hyperparameters of the selected model using Bayesian optimization.

        Args:
            X (np.ndarray): The feature matrix for training.
            y (np.ndarray): The target labels for training.
            metric_optmize (str): The metric to optimize during hyperparameter tuning (default is 'f1_score').
            n_iter (int): The number of iterations for the Bayesian optimization (default is 20).
            cv (int): The number of cross-validation folds (default is 5).
            random_state (int): The random seed for reproducibility (default is 42).

        Returns:
            BayesSearchCV: The fitted Bayesian optimization search object with the best hyperparameters.
        """
        label_encoder = LabelEncoder()  # Initialize the label encoder.
        y = label_encoder.fit_transform(y)  # Encode the target labels to integers.
        
        # Initialize the Bayesian optimization search with the model, parameter grid, and cross-validation settings.
        bayes_search = BayesSearchCV(self.model, 
                                     self.param_grid, 
                                     n_iter=n_iter, 
                                     cv=cv, 
                                     scoring=self._select_scoring(), 
                                     random_state=random_state, 
                                     return_train_score=True, 
                                     refit=metric_optmize,  # Optimize based on the specified metric.
                                     n_jobs=5)  # Use 5 cores for parallel fitting.
        
        bayes_search.fit(X, y)  # Fit the model with the training data and labels.

        return bayes_search  # Return the fitted BayesSearchCV object.
