import os
import warnings
import mlflow
import mlflow.sklearn
import numpy as np
import optuna
import pandas as pd

# pip install optuna-integration[mlflow]
from optuna.integration.mlflow import MLflowCallback
from sklearn.compose import ColumnTransformer, make_column_selector
from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import KFold, cross_validate, train_test_split
from sklearn.neighbors import KNeighborsRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder, RobustScaler, StandardScaler

os.environ["LOKY_MAX_CPU_COUNT"] = "4"  # or 1
warnings.filterwarnings("ignore")

# Load & clean dataset first
data = pd.read_csv("mental_tiredness_score_prediction_dataset.csv")
data = data.drop_duplicates().reset_index(drop=True)

X = data.iloc[:, :-1]
y = data.iloc[:, -1]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Sub-pipelines for numeric and categorical features
num_pipeline = Pipeline(
    steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ]
)

cat_pipeline = Pipeline(
    steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
    ]
)

preprocessor = ColumnTransformer(
    transformers=[
        ("num", num_pipeline, make_column_selector(dtype_include=np.number)),
        ("cat", cat_pipeline, make_column_selector(dtype_exclude=np.number)),
    ],
    remainder="drop",
)

base_pipeline = Pipeline(
    steps=[
        ("Preprocessor", preprocessor),
        ("Model", Ridge()),
    ]
)


def build_scaler(scaler_name: str):
    if scaler_name == "standard":
        return StandardScaler()
    if scaler_name == "minmax":
        return MinMaxScaler()
    return RobustScaler()


def build_model(trial, model_name: str):
    if model_name == "knn":
        # Larger k + uniform weights reduces single-point memorization
        return KNeighborsRegressor(
            n_neighbors=trial.suggest_int("knn_n_neighbors", 7, 35, step=2),
            p=trial.suggest_int("knn_p", 1, 2),
            weights="uniform",
        )
    if model_name == "ridge":
        return Ridge(
            alpha=trial.suggest_float("ridge_alpha", 1e-2, 100.0, log=True)
        )
    if model_name == "rf":
        return RandomForestRegressor(
            n_estimators=trial.suggest_int("rf_n_estimators", 100, 300, step=50),
            max_depth=trial.suggest_int("rf_max_depth", 3, 10),
            min_samples_leaf=trial.suggest_int("rf_min_samples_leaf", 5, 25),
            random_state=42,
            n_jobs=-1,
        )
    return HistGradientBoostingRegressor(
        max_iter=trial.suggest_int("hgb_max_iter", 100, 300, step=50),
        max_depth=trial.suggest_int("hgb_max_depth", 2, 7),
        min_samples_leaf=trial.suggest_int("hgb_min_samples_leaf", 10, 30),
        l2_regularization=trial.suggest_float("hgb_l2", 1e-3, 10.0, log=True),
        learning_rate=trial.suggest_float("hgb_lr", 0.01, 0.15, log=True),
        random_state=42,
    )


def objective(trial):
    scaler_name = trial.suggest_categorical("scaler", ["standard", "minmax", "robust"])
    model_name = trial.suggest_categorical("model", ["knn", "ridge", "rf", "hgb"])

    base_pipeline.set_params(
        Preprocessor__num__scaler=build_scaler(scaler_name),
        Model=build_model(trial, model_name),
    )

    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    cv_res = cross_validate(
        base_pipeline,
        X_train,
        y_train,
        cv=kf,
        scoring="neg_root_mean_squared_error",
        return_train_score=True,
    )

    train_rmse_cv = -cv_res["train_score"].mean()
    val_rmse_cv = -cv_res["test_score"].mean()
    overfit_gap = max(0.0, val_rmse_cv - train_rmse_cv)

    # Light penalty on train-vs-val RMSE gap so trials that overfit train folds are disfavored
    regularized_score = -(val_rmse_cv + 0.25 * overfit_gap)
    trial.set_user_attr("val_rmse_cv", float(val_rmse_cv))
    trial.set_user_attr("train_rmse_cv", float(train_rmse_cv))
    return regularized_score


mlflow_callback = MLflowCallback(
    tracking_uri=mlflow.get_tracking_uri(),
    metric_name="regularized_cv_score",
)

study = optuna.create_study(direction="maximize")
study.optimize(objective, n_trials=80, callbacks=[mlflow_callback])

best_trial = study.best_trial
best_params = best_trial.params
print("Best hyperparameters:", best_params)
print(f"Best CV Val RMSE:   {best_trial.user_attrs['val_rmse_cv']:.4f}")
print(f"Best CV Train RMSE: {best_trial.user_attrs['train_rmse_cv']:.4f}")

# Reconstruct best pipeline cleanly
rebuilt_model = build_model(optuna.trial.FixedTrial(best_params), best_params["model"])
base_pipeline.set_params(
    Preprocessor__num__scaler=build_scaler(best_params["scaler"]),
    Model=rebuilt_model,
)

mlflow.sklearn.autolog()
base_pipeline.fit(X_train, y_train)

y_pred_train = base_pipeline.predict(X_train)
y_pred_test = base_pipeline.predict(X_test)

train_rmse = np.sqrt(mean_squared_error(y_train, y_pred_train))
test_rmse = np.sqrt(mean_squared_error(y_test, y_pred_test))
train_r2 = r2_score(y_train, y_pred_train)
test_r2 = r2_score(y_test, y_pred_test)

print(f"Train RMSE: {train_rmse:.4f} | Train R2: {train_r2:.4f}")
print(f"Test RMSE:  {test_rmse:.4f} | Test R2:  {test_r2:.4f}")

mlflow.log_metric("train_rmse", train_rmse)
mlflow.log_metric("test_rmse", test_rmse)
mlflow.log_metric("train_r2", train_r2)
mlflow.log_metric("test_r2", test_r2)
mlflow.log_metric("rmse_gap", test_rmse - train_rmse)