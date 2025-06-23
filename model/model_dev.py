import optuna

from abc import ABC, abstractmethod
import numpy as np

from sklearn.model_selection import cross_val_score
from sklearn.ensemble import RandomForestRegressor



class Model(ABC):
    """
    Abstract base class for all concrete models
    """

    @abstractmethod
    def train(self, x_train, y_train):
        """
        Trains the model on the given data

        Args:
        x_train: training data
        y_train: target data
        """
        pass

    @abstractmethod
    def objective(self, trial, x_train, x_test, y_train, y_test):
        """
        Optimizes the hyperparameters of the model

        Args:
            trail: Optuna trial object
            x_train: training data
            y_train: target data
            x_test: testing data
            y_test: testing target
        """
        pass


class RandomForestModel(Model):
    """
    Random forest model that implements the model
    """
    # def train(self, x_train, y_train, **kwargs):
    #     model = RandomForestRegressor(**kwargs)
    #     model.fit(x_train, y_train)
    #     return model

    def objective(self, trial, x_train, y_train):

        params = {
                'n_estimators': trial.suggest_int('n_estimators', 500, 500),
                'max_depth': trial.suggest_categorical('max_depth', [None]),
                'min_samples_split': trial.suggest_int('min_samples_split', 20, 20),
                'min_samples_leaf': trial.suggest_int('min_samples_leaf', 3, 3),
                'random_state': 42,
                'n_jobs': -1
            }
    # Cross-validation
        scores = cross_val_score(
            RandomForestRegressor(**params),
            x_train, y_train, cv=5, scoring='r2', n_jobs=-1
        )
        return np.mean(scores)  # Average validation R²

class HyperparameterTuner:
    """
    Class for performing hyperparameter tuning. It used model strategy to perform tuning
    """

    def __init__(self, model, x_train, y_train, x_test, y_test):
        self.model = model
        self.x_train = x_train
        self.y_train = y_train
        self.x_test = x_test
        self.y_test = y_test

    def optimize(self, n_trials=100):
        study = optuna.create_study(direction='maximize')
        study.optimize(lambda trial: self.model.objective(self, trial, self.x_train, self.y_train), n_trials=n_trials)
        return study.best_trial.params

    # 2. After optimization, train final model on full training data
        # best params are: 500,

    def train(self):
        #best_params = self.optimize(n_trials=100)
        best_params = {'n_estimators': 500, 'max_depth': None, 'min_samples_split': 20, 'min_samples_leaf': 3}
        final_model = RandomForestRegressor(**best_params, random_state=42)
        final_model.fit(self.x_test, self.y_test)
        #test_score = final_model.score(self.x_test, self.y_test)
        #print(f'Test R2: {test_score:.4f}')
        return final_model
