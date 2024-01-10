import numpy as np
import pandas as pd

import warnings
import optuna

from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
from sklearn.preprocessing import PowerTransformer
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.linear_model import LinearRegression

from lightgbm import LGBMRegressor


def get_clean_data(
    X: pd.DataFrame, feature_list: list, rename_dict: dict
) -> pd.DataFrame:
    X.loc[
        (X["up_event"] != X["down_event"]) & (X["activity"] == "Nonproduction"),
        "down_event",
    ] = "NoEvent"
    X.loc[
        (X["up_event"] != X["down_event"]) & (X["activity"] == "Nonproduction"),
        "up_event",
    ] = "NoEvent"

    X.loc[
        (X["up_event"] != X["down_event"]) & (X["activity"] == "Input"), "up_event"
    ] = "q"
    X.loc[
        (X["up_event"] != X["down_event"]) & (X["activity"] == "Replace"), "up_event"
    ] = "q"

    X.loc[X["activity"].str.contains("Move From"), "activity"] = "MoveSection"

    X = X.drop(columns=feature_list)
    X = X.rename(columns=rename_dict)

    return X


def rounded_rmse(y, y_pred, **kwargs):
    return mean_squared_error(y, np.round(y_pred * 2) / 2, squared=False)


class FeatureEngineering:
    @staticmethod
    def get_capitalized_letters(X: pd.DataFrame) -> pd.DataFrame:
        X["previous_event_type"] = X["event_type"].shift()
        X["capitalize_letters"] = (
            (X["activity"] == "Input")
            & (X["previous_event_type"] == "Shift")
            & (X["event_type"] == "q")
        )

        X = X.drop(columns=["previous_event_type"])

        return X

    @staticmethod
    def get_temporal_features(X: pd.DataFrame) -> pd.DataFrame:
        X["previous_up_time"] = X["up_time"].shift().fillna(X["down_time"].iloc[0])
        X["time_between_events"] = X["down_time"] - X["previous_up_time"]

        X["cumulative_writing_time"] = (
            X["action_time"] + X["time_between_events"]
        ).cumsum()

        X["warning_issued"] = X["time_between_events"] >= 120000
        X = X.drop(columns=["previous_up_time"])

        return X

    @staticmethod
    def get_cursor_features(X: pd.DataFrame) -> pd.DataFrame:
        X["previous_cursor_position"] = X["cursor_position"].shift().fillna(0)
        X["cursor_move_distance"] = X["cursor_position"] - X["previous_cursor_position"]
        X["cursor_move_distance"] = X["cursor_move_distance"].abs()

        X = X.drop(columns=["previous_cursor_position"])

        return X

    @staticmethod
    def get_word_change_features(X: pd.DataFrame) -> pd.DataFrame:
        X["previous_word_count"] = X["word_count"].shift().fillna(0)
        X["word_count_change"] = X["word_count"] - X["previous_word_count"]
        X["word_count_change"] = X["word_count_change"].abs()

        X = X.drop(columns=["previous_word_count"])

        return X


def calculate_features(unique_dataset):
    feature_list = [
        "id",
        "total_number_of_events",
        "final_number_of_words",
        "number_of_warnings_issued",
        "total_time_taken",
        "total_pause_time",
        "average_pause_length",
        "proportion_pause_time",
        "non_productive_events",
        "input_events",
        "deletion_events",
        "addition_events",
        "replacement_events",
        "string_move_events",
        "number_of_sentences",
        "average_action_time",
        "median_action_time",
        "min_action_time",
        "max_action_time",
        "std_action_time",
        "average_cursor_distance",
        "avg_word_count_btw_events",
        "total_mouse_clicks",
        "total_arrow_btn_clicks",
        "average_time_between_events",
    ]

    data_values = []

    data_values.append(unique_dataset["id"].iloc[0])
    data_values.append(unique_dataset["event_id"].iloc[-1])
    data_values.append(unique_dataset["word_count"].iloc[-1])
    data_values.append(unique_dataset["warning_issued"].sum())
    data_values.append(unique_dataset["cumulative_writing_time"].iloc[-1])
    data_values.append(unique_dataset["time_between_events"].sum())

    data_values.append(unique_dataset["time_between_events"].mean())
    data_values.append(
        unique_dataset["time_between_events"].sum()
        / unique_dataset["cumulative_writing_time"].iloc[-1]
    )

    data_values.extend(
        [
            unique_dataset[unique_dataset["activity"] == "Nonproduction"].shape[0],
            unique_dataset[unique_dataset["activity"] == "Input"].shape[0],
            unique_dataset[unique_dataset["activity"] == "Remove/Cut"].shape[0],
            unique_dataset[unique_dataset["activity"] == "Paste"].shape[0],
            unique_dataset[unique_dataset["activity"] == "Replace"].shape[0],
            unique_dataset[unique_dataset["activity"] == "MoveSection"].shape[0],
        ]
    )

    data_values.append(unique_dataset[unique_dataset["event_type"] == "."].shape[0])
    data_values.append(unique_dataset["action_time"].mean())
    data_values.append(unique_dataset["action_time"].median())
    data_values.append(unique_dataset["action_time"].min())
    data_values.append(unique_dataset["action_time"].max())
    data_values.append(unique_dataset["action_time"].std())

    data_values.append(unique_dataset["cursor_move_distance"].mean())
    data_values.append(unique_dataset["word_count_change"].mean())

    data_values.append(
        unique_dataset[
            (unique_dataset["activity"] == "Nonproduction")
            & (unique_dataset["event_type"].str.contains("click"))
        ].shape[0]
    )
    data_values.append(
        unique_dataset[
            (unique_dataset["activity"] == "Nonproduction")
            & (unique_dataset["event_type"].str.contains("Arrow"))
        ].shape[0]
    )

    data_values.append(unique_dataset["time_between_events"].mean())

    return pd.Series(data_values, index=feature_list)


def create_master_data(input_data: pd.DataFrame, config: dict) -> pd.DataFrame:
    print("Cleaning Train Dataset!")
    cleaned_data = get_clean_data(
        input_data, config["redundant_features"], config["feature_rename"]
    )

    print("Preprocessing Train Data!")
    cleaned_data = cleaned_data.groupby("id", group_keys=False, sort=False).apply(
        FeatureEngineering.get_capitalized_letters
    )
    cleaned_data = cleaned_data.groupby("id", group_keys=False, sort=False).apply(
        FeatureEngineering.get_temporal_features
    )
    cleaned_data = cleaned_data.groupby("id", group_keys=False, sort=False).apply(
        FeatureEngineering.get_cursor_features
    )
    cleaned_data = cleaned_data.groupby("id", group_keys=False, sort=False).apply(
        FeatureEngineering.get_word_change_features
    )

    master_data = (
        cleaned_data.groupby("id").apply(calculate_features).reset_index(drop=True)
    )

    master_data["total_writing_time"] = (
        master_data["total_time_taken"] - master_data["total_pause_time"]
    )

    master_data["proportion_np_events"] = (
        master_data["non_productive_events"] / master_data["total_number_of_events"]
    )
    master_data["proportion_input_events"] = (
        master_data["input_events"] / master_data["total_number_of_events"]
    )
    master_data["proportion_delete_events"] = (
        master_data["deletion_events"] / master_data["total_number_of_events"]
    )
    master_data["proportion_addition_events"] = (
        master_data["addition_events"] / master_data["total_number_of_events"]
    )
    master_data["proportion_replace_events"] = (
        master_data["replacement_events"] / master_data["total_number_of_events"]
    )
    master_data["proportion_moving_events"] = (
        master_data["string_move_events"] / master_data["total_number_of_events"]
    )

    print("Preprocessing Complete!")

    return master_data


class OptunaTuning:
    def __init__(self, X_train, y_train):
        self.X_train = X_train
        self.y_train = y_train

        self.model_dict = {
            "GradientBoostingRegressor": GradientBoostingRegressor,
        }

        self.model_params_func = {
            "GradientBoostingRegressor": self.get_gbr_params,
        }

        self.scalar_dict = {
            "StandardScaler": StandardScaler,
            "RobustScaler": RobustScaler,
            "MinMaxScaler": MinMaxScaler,
        }

    def get_gbr_params(self, trial):
        return {
            "random_state": 0,
            "n_estimators": trial.suggest_int("n_estimators", 100, 1000),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.1),
            "max_depth": trial.suggest_int("max_depth", 3, 10),
            "min_samples_split": trial.suggest_float("min_samples_split", 0.1, 1.0),
            "min_samples_leaf": trial.suggest_float("min_samples_leaf", 0.1, 0.5),
            "subsample": trial.suggest_float("subsample", 0.5, 1.0),
        }

    def objective(self, trial):
        regressor_model_name = "GradientBoostingRegressor"
        scalar_object = self.scalar_dict[
            trial.suggest_categorical("scalar_object", list(self.scalar_dict.keys()))
        ]

        transformer_object = PowerTransformer()
        X = transformer_object.fit_transform(
            scalar_object().fit_transform(self.X_train)
        )

        model_params = self.model_params_func[regressor_model_name](trial)
        regressor_model = self.model_dict[regressor_model_name](**model_params).fit(
            X_train, y_train
        )

        # cv_folds = KFold(n_splits=10, random_state=0, shuffle=True)
        # rmse_scores = cross_val_score(
        #     regressor_model, X, self.y_train, scoring=make_scorer(rounded_rmse, greater_is_better=False), cv=cv_folds)

        y_hat_train = regressor_model.predict(X_train)

        return round(mean_squared_error(y_train, y_hat_train, squared=False), 4)


class RegressorEnsemble(BaseEstimator, RegressorMixin):
    def __init__(self, model_params: dict, models_list: list = None):
        self.models_list = (
            [
                (
                    "gbr",
                    GradientBoostingRegressor(random_state=0, **model_params["gbr"]),
                ),
                ("rfr", RandomForestRegressor(random_state=0)),
                ("lgbm", LGBMRegressor(random_state=0)),
            ]
            if models_list is None
            else models_list
        )

        self.blending_model = None

    def fit(self, X, y=None):
        X_train, X_val, y_train, y_val = train_test_split(
            X, y, test_size=0.33, random_state=0
        )
        meta_X = list()

        for _, model_object in self.models_list:
            model_object.fit(X_train, y_train)
            yhat = model_object.predict(X_val)

            yhat = yhat.reshape(len(yhat), 1)
            meta_X.append(yhat)

        self.blending_model = LinearRegression().fit(np.hstack(meta_X), y_val)

        return self

    def predict(self, X, y=None):
        meta_X = list()

        for _, model_object in self.models_list:
            yhat = model_object.predict(X)

            yhat = yhat.reshape(len(yhat), 1)
            meta_X.append(yhat)

        return self.blending_model.predict(np.hstack(meta_X))


if __name__ == "__main__":
    config = {
        "redundant_features": ["up_event"],
        "feature_rename": {"down_event": "event_type"},
    }

    input_dataset = pd.read_csv("./data/train_logs.csv")
    y_train = pd.read_csv("./data/train_scores.csv")

    master_data = create_master_data(input_data=input_dataset, config=config)
    master_data = pd.merge(master_data, y_train, on="id")
    master_data.to_csv("./data/master_data_v2.csv", index=False)

    master_data = pd.read_csv("./data/master_data_v2.csv")
    master_data = master_data.set_index("id")

    y = master_data["score"]
    X = master_data.drop(columns=["score"])

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.1, random_state=42
    )

    scalar = StandardScaler()
    transformer = PowerTransformer()

    X_train = transformer.fit_transform(scalar.fit_transform(X_train))
    X_test = transformer.transform(scalar.transform(X_test))

    tuning_object = OptunaTuning(X, y)

    study = optuna.create_study(direction="minimize")
    study.optimize(tuning_object.objective, n_trials=100, n_jobs=-1)

    best_params = study.best_params
    print("Best Hyperparameters:", best_params)

    best_mse = study.best_value
    print("Best Root Mean Squared Error:", best_mse)
