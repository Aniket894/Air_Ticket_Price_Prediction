# src/components/model_training.py
import numpy as np
import joblib
from src.logger import logging
from sklearn.linear_model import LinearRegression, Lasso, Ridge, ElasticNet
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, AdaBoostRegressor
from xgboost import XGBRegressor
from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import mean_squared_error, r2_score

def adjusted_r2_score(r2, n, k):
    """
    Calculate the adjusted R² score.
    :param r2: R² score
    :param n: Number of samples
    :param k: Number of features
    :return: Adjusted R² score
    """
    return 1 - (1 - r2) * (n - 1) / (n - k - 1)

def train_models():
    logging.info("Started model training")
    
    # Load transformed datasets
    X_train = np.load('artifacts/X_train.npy')
    y_train = np.load('artifacts/y_train.npy')
    X_test = np.load('artifacts/X_test.npy')
    y_test = np.load('artifacts/y_test.npy')
    logging.info("Loaded transformed datasets")

    # Define models and their parameter grids for randomized search
    models = {
        'LinearRegression': LinearRegression(),
        'Lasso': Lasso(),
        'Ridge': Ridge(),
        'ElasticNet': ElasticNet(),
        'DecisionTreeRegressor': DecisionTreeRegressor(),
        'RandomForestRegressor': RandomForestRegressor(),
        'GradientBoostingRegressor': GradientBoostingRegressor(),
        'AdaBoostRegressor': AdaBoostRegressor(),
        'XGBoostRegressor': XGBRegressor(objective='reg:squarederror')
    }
    
    param_grids = {
        'Lasso': {'alpha': [0.01, 0.1, 1, 10, 100]},
        'Ridge': {'alpha': [0.01, 0.1, 1, 10, 100]},
        'ElasticNet': {'alpha': [0.01, 0.1, 1, 10, 100], 'l1_ratio': [0.1, 0.5, 0.9]},
        'DecisionTreeRegressor': {'max_depth': [3, 5, 7, 9, 11]},
        'RandomForestRegressor': {'n_estimators': [50, 100, 200], 'max_depth': [3, 5, 7, 9, 11]},
        'GradientBoostingRegressor': {'n_estimators': [50, 100, 200], 'learning_rate': [0.01, 0.1, 0.2]},
        'AdaBoostRegressor': {'n_estimators': [50, 100, 200], 'learning_rate': [0.01, 0.1, 0.2]},
        'XGBoostRegressor': {'n_estimators': [50, 100, 200], 'learning_rate': [0.01, 0.1, 0.2], 'max_depth': [3, 5, 7]}
    }
    
    best_model = None
    best_score = float('inf')
    best_r2 = float('-inf')
    best_adj_r2 = float('-inf')

    for model_name, model in models.items():
        logging.info(f"Training model: {model_name}")
        
        param_grid = param_grids.get(model_name, {})
        random_search = RandomizedSearchCV(model, param_grid, n_iter=10, cv=5, scoring='neg_mean_squared_error', n_jobs=-1, random_state=42)
        random_search.fit(X_train, y_train)
        
        best_model_for_current = random_search.best_estimator_
        joblib.dump(best_model_for_current, f'artifacts/{model_name}.pkl')
        logging.info(f"Saved {model_name} model to artifacts folder")

        # Evaluate the model
        y_pred = best_model_for_current.predict(X_test)
        mse = mean_squared_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        adj_r2 = adjusted_r2_score(r2, X_test.shape[0], X_test.shape[1])
        logging.info(f"{model_name} MSE: {mse}")
        logging.info(f"{model_name} R²: {r2}")
        logging.info(f"{model_name} Adjusted R²: {adj_r2}")

        if mse < best_score:
            best_score = mse
            best_r2 = r2
            best_adj_r2 = adj_r2
            best_model = best_model_for_current
    
    joblib.dump(best_model, 'artifacts/best_model.pkl')
    logging.info("Saved the best model to artifacts folder")
    logging.info(f"Best Model MSE: {best_score}")
    logging.info(f"Best Model R²: {best_r2}")
    logging.info(f"Best Model Adjusted R²: {best_adj_r2}")

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    train_models()
