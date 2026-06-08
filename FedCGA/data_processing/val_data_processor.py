
import os
import json
from datetime import datetime

import numpy as np
import pandas as pd
from tqdm import tqdm

from config import get_data_config
from train_data_processor import TrainDataProcessor


class ValidationDataProcessor(TrainDataProcessor):
    def __init__(self, config_path="../configs/data_config.yaml"):
        self.config = get_data_config(config_path)

        project_root = os.path.normpath(
            os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
        )

        self.input_dir = os.path.join(project_root, "data", "raw")
        self.output_dir = os.path.join(project_root, "data", "processed", "val_features")
        self.features_dir = os.path.join(self.output_dir, "features")
        self.metadata_dir = os.path.join(self.output_dir, "metadata")

        os.makedirs(self.features_dir, exist_ok=True)
        os.makedirs(self.metadata_dir, exist_ok=True)

        self.driver_scalers = {}

    def col(self, name):
        return self.config.get_column_name(name)

    def load_data(self, filename="val_set.csv"):
        path = os.path.join(self.input_dir, filename)
        if not os.path.exists(path):
            raise FileNotFoundError(f"Validation data not found: {path}")

        for encoding in ("utf-8", "gbk", "latin-1"):
            try:
                return pd.read_csv(path, encoding=encoding)
            except UnicodeDecodeError:
                continue

        return pd.read_csv(path)

    def prepare_validation_driver_data(self, driver_data):
        available_cols = [
            col for col in self.feature_columns() if col in driver_data.columns
        ]

        if not available_cols:
            return None, None, None

        X = driver_data[available_cols].values
        X = np.nan_to_num(X, nan=0.0, posinf=1e6, neginf=-1e6)

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

        for driver_id in tqdm(df[car_id_col].unique(), desc="Validation drivers"):
            driver_data = df[df[car_id_col] == driver_id].copy()

            if len(driver_data) < min_samples:
                continue

            X, y, feature_names = self.prepare_validation_driver_data(driver_data)
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

        metadata_path = os.path.join(self.metadata_dir, "validation_metadata.csv")
        pd.DataFrame(metadata).to_csv(metadata_path, index=False, encoding="utf-8")

        info = {
            "processing_date": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "dataset_type": "validation",
            "configuration": self.config.name,
            "version": self.config.version,
            "total_drivers_processed": len(results),
            "total_samples_processed": int(
                sum(data["num_samples"] for data in results.values())
            ),
            "feature_configuration": {
                "basic": getattr(self.config, "basic_features_enable", True),
                "trajectory": getattr(self.config, "trajectory_features_enable", True),
                "weather": getattr(self.config, "weather_features_enable", True),
                "location": getattr(self.config, "location_features_enable", True),
                "normalization": getattr(self.config, "normalize_features", True),
            },
        }

        info_path = os.path.join(self.metadata_dir, "processing_info.json")
        with open(info_path, "w", encoding="utf-8") as f:
            json.dump(info, f, ensure_ascii=False, indent=2)

        return info

    def run(self, filename="val_set.csv"):
        df = self.load_data(filename)
        df = self.add_basic_features(df)
        df = self.add_trajectory_features(df)

        results, metadata = self.process_drivers(df)
        if not results:
            raise RuntimeError("No valid validation driver data was generated.")

        info = self.save_results(results, metadata)

        print(
            f"Processed {info['total_drivers_processed']} validation drivers, "
            f"{info['total_samples_processed']} samples."
        )

        return info


def process_validation_set():
    processor = ValidationDataProcessor()
    processor.run()


if __name__ == "__main__":
    process_validation_set()
