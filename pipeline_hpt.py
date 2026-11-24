from pathlib import Path
import os
import time
import warnings
import joblib
import mlflow
import mlflow.sklearn
import numpy as np
import optuna
from optuna.integration.mlflow import MLflowCallback
import pandas as pd
from sklearn.compose import ColumnTransformer, make_column_selector
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import KFold, cross_val_score, train_test_split
from sklearn.neighbors import KNeighborsRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder, RobustScaler, StandardScaler
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor

os.environ["LOKY_MAX_CPU_COUNT"] = "4"
os.environ.setdefault("OMP_NUM_THREADS", "4")
warnings.filterwarnings("ignore")
optuna.logging.set_verbosity(optuna.logging.WARNING)

BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / "mental_tiredness_score_prediction_dataset.csv"
EXPERIMENT_NAME = "MENTAL_TIREDNESS_PL_RUNS"
N_TRIALS = 20
RANDOM_STATE = 42

SCALER_MAP = {"standard": 0, "minmax": 1, "robust": 2}


def get_scaler(scaler_type: str):
    if scaler_type == "standard":
        return StandardScaler()
    if scaler_type == "minmax":
        return MinMaxScaler()
    return RobustScaler()


def build_preprocessor(scaler) -> ColumnTransformer:
    num_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", scaler),
        ]
    )
    cat_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
        ]
    )
    return ColumnTransformer(
        transformers=[
            ("num", num_pipeline, make_column_selector(dtype_include=np.number)),
            ("cat", cat_pipeline, make_column_selector(dtype_exclude=np.number)),
        ],
        remainder="drop",
    )


def build_model(model_name: str, params: dict):
    if model_name == "KNN":
        return KNeighborsRegressor(
            n_neighbors=params["n_neighbors"],
            weights=params["weights"],
            p=params["p"],
        )
    if model_name == "DecisionTree":
        return DecisionTreeRegressor(
            criterion=params["criterion"],
            max_depth=params["max_depth"],
            min_samples_split=params["min_samples_split"],
            min_samples_leaf=params["min_samples_leaf"],
            max_features=params["max_features"],
            random_state=RANDOM_STATE,
        )
    if model_name == "SVR":
        svr_kwargs = {
            "kernel": params["kernel"],
            "C": params["C"],
            "epsilon": params["epsilon"],
        }
        if params["kernel"] in ("rbf", "poly", "sigmoid"):
            svr_kwargs["gamma"] = params["gamma"]
        if params["kernel"] == "poly":
            svr_kwargs["degree"] = params["degree"]
        return SVR(**svr_kwargs)
    if model_name == "Ridge":
        return Ridge(
            alpha=params["alpha"],
            random_state=RANDOM_STATE,
        )
    if model_name == "RandomForest":
        return RandomForestRegressor(
            n_estimators=params["n_estimators"],
            criterion=params["criterion"],
            max_depth=params["max_depth"],
            min_samples_split=params["min_samples_split"],
            min_samples_leaf=params["min_samples_leaf"],
            max_features=params["max_features"],
            bootstrap=params["bootstrap"],
            random_state=RANDOM_STATE,
            n_jobs=-1,
        )
    if model_name == "GradientBoosting":
        return GradientBoostingRegressor(
            n_estimators=params["n_estimators"],
            learning_rate=params["learning_rate"],
            max_depth=params["max_depth"],
            min_samples_split=params["min_samples_split"],
            min_samples_leaf=params["min_samples_leaf"],
            max_features=params["max_features"],
            subsample=params["subsample"],
            random_state=RANDOM_STATE,
        )
    raise ValueError(f"Unsupported model: {model_name}")


def sample_params(model_name: str, trial: optuna.Trial) -> dict:
    params = {
        "scaler_type": trial.suggest_categorical("scaler_type", ["standard", "minmax", "robust"])
    }
    if model_name == "KNN":
        params.update(
            {
                "n_neighbors": trial.suggest_int("n_neighbors", 5, 25, step=2),
                "weights": trial.suggest_categorical("weights", ["uniform", "distance"]),
                "p": trial.suggest_int("p", 1, 2),
            }
        )
    elif model_name == "DecisionTree":
        params.update(
            {
                "criterion": trial.suggest_categorical("criterion", ["squared_error", "friedman_mse"]),
                "max_depth": trial.suggest_int("max_depth", 2, 12),
                "min_samples_split": trial.suggest_int("min_samples_split", 4, 20),
                "min_samples_leaf": trial.suggest_int("min_samples_leaf", 3, 20),
                "max_features": trial.suggest_categorical("max_features", [None, "sqrt", "log2"]),
            }
        )
    elif model_name == "SVR":
        kernel = trial.suggest_categorical("kernel", ["linear", "rbf", "poly", "sigmoid"])
        params.update(
            {
                "kernel": kernel,
                "C": trial.suggest_float("C", 1e-2, 1e2, log=True),
                "epsilon": trial.suggest_float("epsilon", 0.05, 2.0),
            }
        )
        if kernel in ("rbf", "poly", "sigmoid"):
            params["gamma"] = trial.suggest_float("gamma", 1e-4, 1e-1, log=True)
        if kernel == "poly":
            params["degree"] = trial.suggest_int("degree", 2, 4)
    elif model_name == "Ridge":
        params.update(
            {
                "alpha": trial.suggest_float("alpha", 1e-3, 1e3, log=True),
            }
        )
    elif model_name == "RandomForest":
        params.update(
            {
                "n_estimators": trial.suggest_int("n_estimators", 100, 350, step=50),
                "criterion": trial.suggest_categorical("criterion", ["squared_error", "friedman_mse"]),
                "max_depth": trial.suggest_int("max_depth", 3, 14),
                "min_samples_split": trial.suggest_int("min_samples_split", 4, 20),
                "min_samples_leaf": trial.suggest_int("min_samples_leaf", 3, 20),
                "max_features": trial.suggest_categorical("max_features", ["sqrt", "log2", None]),
                "bootstrap": trial.suggest_categorical("bootstrap", [True, False]),
            }
        )
    elif model_name == "GradientBoosting":
        params.update(
            {
                "n_estimators": trial.suggest_int("n_estimators", 100, 350, step=50),
                "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.2, log=True),
                "max_depth": trial.suggest_int("max_depth", 2, 6),
                "min_samples_split": trial.suggest_int("min_samples_split", 4, 20),
                "min_samples_leaf": trial.suggest_int("min_samples_leaf", 3, 20),
                "max_features": trial.suggest_categorical("max_features", [None, "sqrt", "log2"]),
                "subsample": trial.suggest_float("subsample", 0.6, 1.0),
            }
        )
    return params


def make_objective(model_name: str, X_train: pd.DataFrame, y_train: pd.Series, cv: KFold):
    def objective(trial: optuna.Trial) -> float:
        params = sample_params(model_name, trial)
        pipe = Pipeline(
            steps=[
                ("Preprocessor", build_preprocessor(get_scaler(params["scaler_type"]))),
                ("Model", build_model(model_name, params)),
            ]
        )
        scores = cross_val_score(
            pipe,
            X_train,
            y_train,
            scoring="neg_root_mean_squared_error",
            cv=cv,
            n_jobs=-1,
        )
        return float(scores.mean())

    return objective


def main():
    data = pd.read_csv(DATA_PATH).drop_duplicates().reset_index(drop=True)
    X = data.iloc[:, :-1]
    y = data.iloc[:, -1]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=RANDOM_STATE
    )

    cv = KFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    model_names = ["KNN", "DecisionTree", "SVR", "Ridge", "RandomForest", "GradientBoosting"]
    model_id_map = {name: idx for idx, name in enumerate(model_names)}

    mlflow.set_experiment(EXPERIMENT_NAME)
    results = {}

    for model_name in model_names:
        print(f"\n--- Optimizing {model_name} ---")
        mlflow_cb = MLflowCallback(
            tracking_uri=mlflow.get_tracking_uri(),
            metric_name="cv_neg_rmse",
            mlflow_kwargs={"nested": True},
        )

        study = optuna.create_study(
            direction="maximize",
            sampler=optuna.samplers.TPESampler(seed=RANDOM_STATE),
        )
        obj_fn = make_objective(model_name, X_train, y_train, cv)

        start_fit = time.time()
        study.optimize(obj_fn, n_trials=N_TRIALS, callbacks=[mlflow_cb])
        fit_time = time.time() - start_fit

        best_params = study.best_params
        best_cv_rmse = float(-study.best_value)

        best_pipe = Pipeline(
            steps=[
                ("Preprocessor", build_preprocessor(get_scaler(best_params["scaler_type"]))),
                ("Model", build_model(model_name, best_params)),
            ]
        )
        best_pipe.fit(X_train, y_train)

        y_train_pred = best_pipe.predict(X_train)
        start_test = time.time()
        y_test_pred = best_pipe.predict(X_test)
        test_time = time.time() - start_test

        train_rmse = float(np.sqrt(mean_squared_error(y_train, y_train_pred)))
        test_rmse = float(np.sqrt(mean_squared_error(y_test, y_test_pred)))
        train_r2 = float(r2_score(y_train, y_train_pred))
        test_r2 = float(r2_score(y_test, y_test_pred))
        rmse_gap = float(test_rmse - train_rmse)

        model_path = BASE_DIR / f"{model_name}_final_model.pkl"
        joblib.dump(best_pipe, model_path)
        model_size = int(os.path.getsize(model_path))

        mlflow.log_params({f"best_{k}": v for k, v in best_params.items()})
        mlflow.log_metrics(
            {
                "model_id": model_id_map[model_name],
                "scaler_id": SCALER_MAP[best_params["scaler_type"]],
                "best_cv_rmse": best_cv_rmse,
                "train_rmse": train_rmse,
                "test_rmse": test_rmse,
                "train_r2": train_r2,
                "test_r2": test_r2,
                "rmse_gap": rmse_gap,
                "train_time": fit_time,
                "test_time": test_time,
                "model_size": model_size,
            }
        )
        mlflow.sklearn.log_model(
            best_pipe,
            name=f"{model_name}_tiredness_model",
            serialization_format="cloudpickle",
        )
        model_path.unlink(missing_ok=True)
        mlflow.end_run()

        results[model_name] = {
            "best_cv_rmse": best_cv_rmse,
            "train_rmse": train_rmse,
            "test_rmse": test_rmse,
            "train_r2": train_r2,
            "test_r2": test_r2,
            "rmse_gap": rmse_gap,
            "fit_time": fit_time,
            "test_time": test_time,
            "model_size_bytes": model_size,
        }
        print(
            f"{model_name} | CV RMSE: {best_cv_rmse:.4f} | "
            f"Train RMSE: {train_rmse:.4f} (R2: {train_r2:.4f}) | "
            f"Test RMSE: {test_rmse:.4f} (R2: {test_r2:.4f}) | Time: {fit_time:.2f}s"
        )

    print("\n--- Summary (Sorted by Test RMSE) ---")
    ranked = sorted(results.items(), key=lambda item: item[1]["test_rmse"])
    for model_name, res in ranked:
        print(
            f"{model_name:16s} | CV RMSE: {res['best_cv_rmse']:.4f} | "
            f"Train RMSE: {res['train_rmse']:.4f} | Test RMSE: {res['test_rmse']:.4f} | "
            f"Train R2: {res['train_r2']:.4f} | Test R2: {res['test_r2']:.4f} | "
            f"Gap: {res['rmse_gap']:+.4f} | Fit: {res['fit_time']:.2f}s | "
            f"Size: {res['model_size_bytes']} B"
        )


if __name__ == "__main__":
    main()