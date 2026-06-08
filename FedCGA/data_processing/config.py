import os
import yaml
import json
from pathlib import Path
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Union
import warnings


@dataclass
class DataConfig:
    name: str
    version: str
    description: str
    date_format: str
    datetime_format: str

    column_mapping: Dict[str, str]

    basic_features_enable: bool
    trajectory_features_enable: bool
    weather_features_enable: bool
    location_features_enable: bool

    min_samples_per_driver: int
    test_size: float
    random_seed: int
    normalize_features: bool
    price_coefficient_range: Dict[str, float]

    haversine_radius: float
    grid_size: float

    outlier_handling_enable: bool
    z_score_threshold: float
    iqr_multiplier: float

    save_driver_features: bool
    save_metadata: bool
    save_scalers: bool
    compress_features: bool
    feature_format: str
    metadata_format: str
    feature_dir: str
    metadata_dir: str

    basic_feature_names: List[str] = field(default_factory=list)
    trajectory_feature_names: List[str] = field(default_factory=list)
    weather_feature_names: List[str] = field(default_factory=list)
    location_feature_names: List[str] = field(default_factory=list)

    def get_all_feature_names(self) -> List[str]:
        all_features = []
        if self.basic_features_enable:
            all_features.extend(self.basic_feature_names)
        if self.trajectory_features_enable:
            all_features.extend(self.trajectory_feature_names)
        if self.weather_features_enable:
            all_features.extend(self.weather_feature_names)
        if self.location_features_enable:
            all_features.extend(self.location_feature_names)
        return all_features

    def get_column_name(self, logical_name: str) -> str:
        if logical_name in self.column_mapping:
            return self.column_mapping[logical_name]
        else:
            warnings.warn(f"逻辑列名 '{logical_name}' 未在配置中定义，使用逻辑名称作为列名")
            return logical_name

    def get_all_required_columns(self) -> List[str]:
        required_columns = set()

        required_columns.update([
            self.get_column_name('car_id'),
            self.get_column_name('date'),
            self.get_column_name('target')
        ])

        if self.basic_features_enable:
            required_columns.update([
                self.get_column_name('start_time'),
                self.get_column_name('end_time')
            ])

        if self.basic_features_enable or self.location_features_enable:
            required_columns.update([
                self.get_column_name('start_lng'),
                self.get_column_name('start_lat'),
                self.get_column_name('end_lng'),
                self.get_column_name('end_lat')
            ])

        if self.trajectory_features_enable:
            required_columns.add(self.get_column_name('trajectory'))

        if self.weather_features_enable:
            for i in range(6):
                weather_key = f'weather_{i}'
                if weather_key in self.column_mapping:
                    required_columns.add(self.get_column_name(weather_key))

        return list(required_columns)

    def to_dict(self) -> Dict[str, Any]:
        return {
            'name': self.name,
            'version': self.version,
            'description': self.description,
            'column_mapping': self.column_mapping,
            'feature_config': {
                'basic_features': {
                    'enable': self.basic_features_enable,
                    'features': self.basic_feature_names
                },
                'trajectory_features': {
                    'enable': self.trajectory_features_enable,
                    'features': self.trajectory_feature_names
                },
                'weather_features': {
                    'enable': self.weather_features_enable,
                    'features': self.weather_feature_names
                },
                'location_features': {
                    'enable': self.location_features_enable,
                    'features': self.location_feature_names
                }
            },
            'processing_params': {
                'min_samples_per_driver': self.min_samples_per_driver,
                'test_size': self.test_size,
                'random_seed': self.random_seed,
                'normalize_features': self.normalize_features,
                'price_coefficient_range': self.price_coefficient_range
            }
        }

    def save(self, filepath: str):
        config_dir = os.path.dirname(filepath)
        os.makedirs(config_dir, exist_ok=True)

        with open(filepath, 'w', encoding='utf-8') as f:
            yaml.dump(self.to_dict(), f, default_flow_style=False, allow_unicode=True, indent=2)

        print(f"配置已保存到: {filepath}")


class ConfigLoader:

    def __init__(self, config_dir: str = "configs"):
        self.config_dir = config_dir
        self.data_config_path = os.path.join(config_dir, "data_config.yaml")
        self.model_config_path = os.path.join(config_dir, "model_config.yaml")

    def load_data_config(self, config_path: Optional[str] = None) -> DataConfig:
        if config_path is None:
            config_path = self.data_config_path

        if not os.path.exists(config_path):
            raise FileNotFoundError(f"配置文件不存在: {config_path}")

        print(f"加载配置文件: {config_path}")
        with open(config_path, 'r', encoding='utf-8') as f:
            config_dict = yaml.safe_load(f)

        data_info = config_dict.get('data_info', {})
        column_mapping = config_dict.get('column_mapping', {})
        feature_config = config_dict.get('feature_config', {})
        processing_params = config_dict.get('processing_params', {})
        output_config = config_dict.get('output_config', {})

        traj_params = processing_params.get('trajectory_processing', {})

        outlier_params = processing_params.get('outlier_handling', {})

        config = DataConfig(
            name=data_info.get('name', ''),
            version=data_info.get('version', '1.0'),
            description=data_info.get('description', ''),
            date_format=data_info.get('date_format', '%Y/%m/%d'),
            datetime_format=data_info.get('datetime_format', '%Y/%m/%d %H:%M'),

            column_mapping=column_mapping,

            basic_features_enable=feature_config.get('basic_features', {}).get('enable', True),
            trajectory_features_enable=feature_config.get('trajectory_features', {}).get('enable', True),
            weather_features_enable=feature_config.get('weather_features', {}).get('enable', True),
            location_features_enable=feature_config.get('location_features', {}).get('enable', True),

            min_samples_per_driver=processing_params.get('min_samples_per_driver', 20),
            test_size=processing_params.get('test_size', 0.2),
            random_seed=processing_params.get('random_seed', 42),
            normalize_features=processing_params.get('normalize_features', True),
            price_coefficient_range=processing_params.get('price_coefficient_range', {'min': 0.85, 'max': 1.3}),

            haversine_radius=traj_params.get('haversine_radius', 6371),
            grid_size=traj_params.get('grid_size', 0.1),

            outlier_handling_enable=outlier_params.get('enable', True),
            z_score_threshold=outlier_params.get('z_score_threshold', 3.0),
            iqr_multiplier=outlier_params.get('iqr_multiplier', 1.5),

            save_driver_features=output_config.get('save_driver_features', True),
            save_metadata=output_config.get('save_metadata', True),
            save_scalers=output_config.get('save_scalers', True),
            compress_features=output_config.get('compress_features', True),
            feature_format=output_config.get('feature_format', 'npz'),
            metadata_format=output_config.get('metadata_format', 'csv'),
            feature_dir=output_config.get('feature_dir', '特征'),
            metadata_dir=output_config.get('metadata_dir', '元数据'),

            basic_feature_names=feature_config.get('basic_features', {}).get('features', []),
            trajectory_feature_names=feature_config.get('trajectory_features', {}).get('features', []),
            weather_feature_names=feature_config.get('weather_features', {}).get('features', []),
            location_feature_names=feature_config.get('location_features', {}).get('features', [])
        )

        print(f"配置加载成功: {config.name} v{config.version}")
        print(f"列名映射: {len(config.column_mapping)} 个映射关系")
        print(f"特征配置: 基础{config.basic_features_enable}, 轨迹{config.trajectory_features_enable}, "
              f"天气{config.weather_features_enable}, 位置{config.location_features_enable}")

        return config

    def load_model_config(self, config_path: Optional[str] = None) -> Dict[str, Any]:
        if config_path is None:
            config_path = self.model_config_path

        if not os.path.exists(config_path):
            raise FileNotFoundError(f"模型配置文件不存在: {config_path}")

        with open(config_path, 'r', encoding='utf-8') as f:
            config_dict = yaml.safe_load(f)

        return config_dict

    def create_default_data_config(self, output_path: Optional[str] = None):
        if output_path is None:
            output_path = self.data_config_path

        default_config = {
            'data_info': {
                'name': '网约车订单数据',
                'version': '1.0',
                'description': '用于联邦学习定价模型的订单数据',
                'date_format': '%Y/%m/%d',
                'datetime_format': '%Y/%m/%d %H:%M'
            },
            'column_mapping': {
                'car_id': 'car_id',
                'date': 'date',
                'target': 'dp_cur',
                'start_time': 'trip_start_time',
                'end_time': 'trip_end_time',
                'day_type': 'day_dayty',
                'month': 'month',
                'day': 'day',
                'start_lng': 'slo',
                'start_lat': 'sla',
                'end_lng': 'elo',
                'end_lat': 'ela',
                'trajectory': 'traj',
                'weather_0': 'weather_0',
                'weather_1': 'weather_1',
                'weather_2': 'weather_2',
                'weather_3': 'weather_3',
                'weather_4': 'weather_4',
                'weather_5': 'weather_5'
            }
        }

        config_dir = os.path.dirname(output_path)
        os.makedirs(config_dir, exist_ok=True)

        with open(output_path, 'w', encoding='utf-8') as f:
            yaml.dump(default_config, f, default_flow_style=False, allow_unicode=True, indent=2)

        print(f"默认配置文件已创建: {output_path}")
        return output_path


def get_data_config(config_path: Optional[str] = None) -> DataConfig:
    loader = ConfigLoader()
    return loader.load_data_config(config_path)


def get_model_config(config_path: Optional[str] = None) -> Dict[str, Any]:
    loader = ConfigLoader()
    return loader.load_model_config(config_path)
