import os
import ast
import json
from datetime import datetime

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm

from config import get_data_config


class TrainDataProcessor:
    def __init__(self, config_path="../configs/data_config.yaml"):
        self.config = get_data_config(config_path)

        project_root = os.path.normpath(
            os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
        )

        self.input_dir = os.path.join(project_root, "data", "raw")
        self.output_dir = os.path.join(project_root, "data", "processed", "train_features")
        self.features_dir = os.path.join(self.output_dir, "features")
        self.metadata_dir = os.path.join(self.output_dir, "metadata")

        os.makedirs(self.features_dir, exist_ok=True)
        os.makedirs(self.metadata_dir, exist_ok=True)

        self.driver_scalers = {}

    def col(self, name):
        return self.config.get_column_name(name)

    def load_data(self, filename="train_set.csv"):
        path = os.path.join(self.input_dir, filename)
        if not os.path.exists(path):
            raise FileNotFoundError(f"Training data not found: {path}")

        for encoding in ("utf-8", "gbk", "latin-1"):
            try:
                return pd.read_csv(path, encoding=encoding)
            except UnicodeDecodeError:
                continue

        return pd.read_csv(path)

    def haversine_distance(self, lon1, lat1, lon2, lat2):
        radius = self.config.haversine_radius

        lon1, lat1, lon2, lat2 = map(np.radians, [lon1, lat1, lon2, lat2])
        dlon = lon2 - lon1
        dlat = lat2 - lat1

        a = (
            np.sin(dlat / 2) ** 2
            + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2) ** 2
        )
        c = 2 * np.arcsin(np.sqrt(a))

        return radius * c

    def add_basic_features(self, df):
        start_time_col = self.col("start_time")
        end_time_col = self.col("end_time")
        date_col = self.col("date")
        start_lng_col = self.col("start_lng")
        start_lat_col = self.col("start_lat")
        end_lng_col = self.col("end_lng")
        end_lat_col = self.col("end_lat")

        df["trip_start_time"] = pd.to_datetime(df[start_time_col], errors="coerce")
        df["trip_end_time"] = pd.to_datetime(df[end_time_col], errors="coerce")

        df["start_hour"] = df["trip_start_time"].dt.hour.fillna(0)
        df["hour_sin"] = np.sin(2 * np.pi * df["start_hour"] / 24)
        df["hour_cos"] = np.cos(2 * np.pi * df["start_hour"] / 24)

        df["duration"] = (
            df["trip_end_time"] - df["trip_start_time"]
        ).dt.total_seconds() / 60
        df["duration"] = df["duration"].fillna(0).clip(lower=0)

        df["date_dt"] = pd.to_datetime(df[date_col], errors="coerce")
        df["day_of_week"] = df["date_dt"].dt.dayofweek.fillna(0)
        df["is_weekend"] = df["day_of_week"].isin([5, 6]).astype(int)

        df["straight_distance"] = self.haversine_distance(
            df[start_lng_col],
            df[start_lat_col],
            df[end_lng_col],
            df[end_lat_col],
        )

        df["speed"] = df["straight_distance"] / (df["duration"] / 60 + 1e-5)
        df["direction"] = np.arctan2(
            df[end_lat_col] - df[start_lat_col],
            df[end_lng_col] - df[start_lng_col],
        )

        if self.config.weather_features_enable:
            for i in range(6):
                key = f"weather_{i}"
                if key in self.config.column_mapping:
                    source_col = self.col(key)
                    if source_col in df.columns:
                        df[f"{key}_norm"] = df[source_col]

        return df

    def add_trajectory_features(self, df):
        if not self.config.trajectory_features_enable:
            return df

        trajectory_col = self.col("trajectory")
        if trajectory_col not in df.columns:
            return df

        features = []

        for idx, value in tqdm(
            enumerate(df[trajectory_col]),
            total=len(df),
            desc="Trajectory features",
        ):
            traj_length = df.iloc[idx].get("straight_distance", 0)
            complexity = 1.0
            num_points = 0

            try:
                points = ast.literal_eval(value) if isinstance(value, str) else []

                if len(points) >= 2:
                    traj_length = 0.0
                    for i in range(1, len(points)):
                        lon1, lat1 = points[i - 1]
                        lon2, lat2 = points[i]
                        traj_length += self.haversine_distance(lon1, lat1, lon2, lat2)

                    straight_distance = df.iloc[idx].get("straight_distance", 0)
                    complexity = (
                        traj_length / straight_distance
                        if straight_distance > 0
                        else 1.0
                    )
                    num_points = len(points)

            except Exception:
                pass

            features.append(
                {
                    "traj_length": traj_length,
                    "traj_complexity": complexity,
                    "num_points": num_points,
                }
            )

        traj_df = pd.DataFrame(features)
        for col in traj_df.columns:
            df[f"traj_{col}"] = traj_df[col].values

        return df

    def feature_columns(self):
        columns = []

        if self.config.basic_features_enable:
            columns.extend(
                [
                    "hour_sin",
                    "hour_cos",
                    "duration",
                    "start_hour",
                    "day_of_week",
                    "is_weekend",
                    "straight_distance",
                    "speed",
                    "direction",
                ]
            )

        if self.config.trajectory_features_enable:
            columns.extend(["traj_length", "traj_complexity", "num_points"])

        if self.config.weather_features_enable:
            columns.extend([f"weather_{i}_norm" for i in range(6)])

        if self.config.location_features_enable:
            columns.extend(
                [
                    self.col("start_lng"),
                    self.col("start_lat"),
                    self.col("end_lng"),
                    self.col("end_lat"),
                ]
            )

        return columns

    def prepare_driver_data(self, driver_data, driver_id):
        available_cols = [
            col for col in self.feature_columns() if col in driver_data.columns
        ]

        if not available_cols:
            return None, None, None

        X = driver_data[available_cols].values
        X = np.nan_to_num(X, nan=0.0, posinf=1e6, neginf=-1e6)

        if self.config.normalize_features:
            scaler = self.driver_scalers.get(driver_id)
            if scaler is None:
                scaler = StandardScaler()
                scaler.fit(X)
                self.driver_scalers[driver_id] = scaler
            X = scaler.transform(X)

        target_col = self.col("target")
        if target_col not in driver_data.columns:
            return None, None, None

        y = driver_data[target_col].values.reshape(-1, 1)

        return X.astype(np.float32), y.astype(np.float32).flatten(), available_cols

    def process_drivers(self, df):
        car_id_col = self.col("car_id")
        date_col = self.col("date")
        min_samples = getattr(self.config, "min_samples_per_driver", 10)

        results = {}
        metadata = []

        for driver_id in tqdm(df[car_id_col].unique(), desc="Drivers"):
            driver_data = df[df[car_id_col] == driver_id].copy()

            if len(driver_data) < min_samples:
                continue

            X, y, feature_names = self.prepare_driver_data(driver_data, driver_id)
            if X is None or y is None:
                continue

            results[driver_id] = {
                "X": X,
                "y": y,
                "num_samples": len(driver_data),
                "feature_names": feature_names,
            }

            metadata.append(
                {
                    "driver_id": driver_id,
                    "num_samples": len(driver_data),
                    "num_features": X.shape[1],
                    "target_mean": float(y.mean()),
                    "target_std": float(y.std()),
                    "target_min": float(y.min()),
                    "target_max": float(y.max()),
                    "start_date": driver_data[date_col].min(),
                    "end_date": driver_data[date_col].max(),
                }
            )

        return results, metadata

    def save_results(self, results, metadata):
        for driver_id, data in results.items():
            path = os.path.join(self.features_dir, f"driver_{driver_id}.npz")
            np.savez_compressed(path, X=data["X"], y=data["y"])

        metadata_path = os.path.join(self.metadata_dir, "training_metadata.csv")
        pd.DataFrame(metadata).to_csv(metadata_path, index=False, encoding="utf-8")

        info = {
            "processing_date": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "configuration": self.config.name,
            "version": self.config.version,
            "total_drivers_processed": len(results),
            "total_samples_processed": int(
                sum(data["num_samples"] for data in results.values())
            ),
            "feature_configuration": {
                "basic": self.config.basic_features_enable,
                "trajectory": self.config.trajectory_features_enable,
                "weather": self.config.weather_features_enable,
                "location": self.config.location_features_enable,
                "normalization": self.config.normalize_features,
            },
        }

        info_path = os.path.join(self.metadata_dir, "processing_info.json")
        with open(info_path, "w", encoding="utf-8") as f:
            json.dump(info, f, ensure_ascii=False, indent=2)

        return info

    def run(self, filename="train_set.csv"):
        df = self.load_data(filename)
        df = self.add_basic_features(df)
        df = self.add_trajectory_features(df)

        results, metadata = self.process_drivers(df)
        if not results:
            raise RuntimeError("No valid driver data was generated.")

        info = self.save_results(results, metadata)

        print(
            f"Processed {info['total_drivers_processed']} drivers, "
            f"{info['total_samples_processed']} samples."
        )

        return info


def process_training_set():
    processor = TrainDataProcessor()
    processor.run()


if __name__ == "__main__":
    process_training_set()