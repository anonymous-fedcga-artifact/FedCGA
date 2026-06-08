#!/usr/bin/env python3
import os
import sys
import yaml
import json
import numpy as np
import torch
import torch.nn as nn
import time
import random
import psutil
from datetime import datetime
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.normpath(os.path.join(current_dir, '..'))
sys.path.insert(0, project_root)

from pricing_model.pricing_model import create_pricing_model


class DeviceManager:

    def __init__(self):
        if torch.cuda.is_available():
            self.server_device = torch.device('cuda')
            self.gpu_name = torch.cuda.get_device_name(0)
        else:
            self.server_device = torch.device('cpu')
            self.gpu_name = "None"

        self.client_device = torch.device('cpu')

        self.client_compute_time = 0.0
        self.server_compute_time = 0.0
        self.data_transfer_time = 0.0
        self.client_count = 0
        self.round_count = 0

        self._log_device_info()

    def _log_device_info(self):
        print(f"🎯🎯 设备配置信息:")
        print(f"   服务器设备: {self.server_device} ({self.gpu_name})")
        print(f"   客户端设备: {self.client_device}")
        print(f"   设备分离模式: 客户端CPU + 服务器GPU")

    def record_client_time(self, time_taken):
        self.client_compute_time += time_taken
        self.client_count += 1

    def record_server_time(self, time_taken):
        self.server_compute_time += time_taken
        self.round_count += 1

    def record_transfer_time(self, time_taken):
        self.data_transfer_time += time_taken

    def get_compute_statistics(self):
        return {
            'total_client_time': self.client_compute_time,
            'total_server_time': self.server_compute_time,
            'total_transfer_time': self.data_transfer_time,
            'avg_client_time': self.client_compute_time / max(1, self.client_count),
            'avg_server_time': self.server_compute_time / max(1, self.round_count),
            'total_clients': self.client_count,
            'total_rounds': self.round_count,
            'client_device_ratio': self.client_compute_time / max(0.001, self.server_compute_time)
        }


class FedAvgTrainingSystem:

    def __init__(self, config_path=None):
        self.script_dir = os.path.dirname(os.path.abspath(__file__))
        self.project_root = os.path.normpath(os.path.join(self.script_dir, '..'))

        self.config_path = self._resolve_config_path(config_path)
        self.config = self._load_config()

        self.device_manager = DeviceManager()

        self._setup_output_directories()

        print("✅✅ FedAvg训练系统初始化完成（设备分离模式）")

    def _resolve_config_path(self, config_path):
        if config_path is None:
            config_path = os.path.join(self.project_root, 'configs', 'fedavg_config.yaml')
        elif not os.path.isabs(config_path):
            config_path = os.path.join(self.project_root, config_path)

        if not os.path.exists(config_path):
            raise FileNotFoundError(f"❌❌ 配置文件不存在: {config_path}")

        return os.path.normpath(config_path)

    def _load_config(self):
        print(f"✅✅ 加载配置文件: {self.config_path}")
        with open(self.config_path, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
        return config.get('fedavg', {})

    def _setup_output_directories(self):
        model_type = self.config.get('model_type', 'mlp')
        algorithm = self.config.get('algorithm', 'fedavg')

        self.output_dir = os.path.join(
            self.project_root, 'outputs',
            f'{algorithm}_{model_type}_device_separated'
        )
        self.models_dir = os.path.join(self.output_dir, 'global_models')
        self.results_dir = os.path.join(self.output_dir, 'training_results')
        self.logs_dir = os.path.join(self.output_dir, 'training_logs')

        for directory in [self.models_dir, self.results_dir, self.logs_dir]:
            os.makedirs(directory, exist_ok=True)

        print(f"📁📁 输出目录: {self.output_dir}")

    def load_client_data(self, client_id, data_type='train'):
        data_path = os.path.join(
            self.project_root, 'data', 'processed', f'{data_type}_features',
            'features', f'driver_{client_id}.npz'
        )

        if not os.path.exists(data_path):
            print(f"❌❌ 客户端数据不存在: {data_path}")
            return None, None

        try:
            data = np.load(data_path)
            X = torch.FloatTensor(data['X'])
            y = torch.FloatTensor(data['y'])
            return X, y
        except Exception as e:
            print(f"❌❌ 加载客户端 {client_id} 数据失败: {e}")
            return None, None

    def get_all_client_ids(self, data_type='train'):
        features_dir = os.path.join(
            self.project_root, 'data', 'processed', f'{data_type}_features', 'features'
        )

        if not os.path.exists(features_dir):
            return []

        client_files = [f for f in os.listdir(features_dir)
                        if f.startswith('driver_') and f.endswith('.npz')]
        client_ids = [f.replace('driver_', '').replace('.npz', '') for f in client_files]

        print(f"📊📊 {data_type}集客户端数量: {len(client_ids)}")
        return client_ids

    def create_model(self):
        model_type = self.config.get('model_type', 'mlp')
        model_params = self.config.get('model_params', {}).get(model_type, {})

        try:
            model = create_pricing_model(model_type, **model_params)
            model = model.to(self.device_manager.server_device)
            print(f"🔧🔧 创建{model_type}模型成功（设备: {self.device_manager.server_device}）")
            return model
        except Exception as e:
            print(f"❌❌ 创建模型失败: {e}")
            return None

    def client_local_train(self, global_model, client_id, local_epochs=None):
        if local_epochs is None:
            local_epochs = self.config.get('local_epochs', 5)

        client_start_time = time.time()

        X, y = self.load_client_data(client_id, 'train')
        if X is None:
            return None

        client_model = create_pricing_model(
            self.config.get('model_type', 'mlp'),
            **self.config.get('model_params', {}).get(self.config.get('model_type', 'mlp'), {})
        )

        transfer_start = time.time()
        cpu_global_state = {k: v.cpu() for k, v in global_model.state_dict().items()}
        client_model.load_state_dict(cpu_global_state)
        client_model = client_model.to(self.device_manager.client_device)
        transfer_time = time.time() - transfer_start
        self.device_manager.record_transfer_time(transfer_time)

        client_model.train()

        optimizer = torch.optim.Adam(
            client_model.parameters(),
            lr=self.config.get('learning_rate', 0.001)
        )
        criterion = nn.MSELoss()

        from torch.utils.data import TensorDataset, DataLoader
        dataset = TensorDataset(
            X.to(self.device_manager.client_device),
            y.to(self.device_manager.client_device)
        )
        train_loader = DataLoader(dataset, batch_size=32, shuffle=True)

        epoch_losses = []
        for epoch in range(local_epochs):
            epoch_loss = 0.0
            for batch_X, batch_y in train_loader:
                optimizer.zero_grad()
                outputs = client_model(batch_X).squeeze()
                loss = criterion(outputs, batch_y)
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item()

            avg_loss = epoch_loss / len(train_loader)
            epoch_losses.append(avg_loss)

        client_time = time.time() - client_start_time
        self.device_manager.record_client_time(client_time)

        return {
            'model_params': client_model.state_dict(),
            'num_samples': len(X),
            'client_id': client_id,
            'train_loss': epoch_losses[-1],
            'compute_time': client_time
        }

    def fedavg_aggregate(self, client_updates):
        if not client_updates:
            raise ValueError("❌❌ 没有客户端更新可聚合")

        server_start_time = time.time()

        total_samples = sum(update['num_samples'] for update in client_updates)
        averaged_params = {}
        first_update = client_updates[0]['model_params']

        for param_name in first_update:
            weighted_sum = torch.zeros_like(
                first_update[param_name],
                device=self.device_manager.server_device
            )

            for update in client_updates:
                weight = update['num_samples'] / total_samples
                param_data = update['model_params'][param_name]
                if param_data.device != self.device_manager.server_device:
                    param_data = param_data.to(self.device_manager.server_device)
                weighted_sum += param_data * weight

            averaged_params[param_name] = weighted_sum

        server_time = time.time() - server_start_time
        self.device_manager.record_server_time(server_time)

        return averaged_params

    def evaluate_model(self, model, client_ids, data_type='val'):
        if not client_ids:
            return None

        all_predictions = []
        all_targets = []
        evaluated_clients = 0

        original_device = next(model.parameters()).device
        model = model.cpu()

        with torch.no_grad():
            for client_id in client_ids:
                X, y = self.load_client_data(client_id, data_type)
                if X is not None:
                    X = X.cpu()
                    predictions = model(X).squeeze().numpy()
                    all_predictions.extend(predictions)
                    all_targets.extend(y.numpy())
                    evaluated_clients += 1

        model = model.to(original_device)

        if len(all_predictions) == 0:
            return None

        y_true = np.array(all_targets)
        y_pred = np.array(all_predictions)

        rmse = np.sqrt(mean_squared_error(y_true, y_pred))
        mae = mean_absolute_error(y_true, y_pred)

        range_error_percent = rmse / 0.6 * 100

        mape_percent = 0.0
        if np.all(y_true != 0):
            mape_percent = np.mean(np.abs((y_true - y_pred) / y_true)) * 100

        return {
            'avg_rmse': rmse,
            'avg_mae': mae,
            'range_error_percent': range_error_percent,
            'mape_percent': mape_percent,
            'clients_evaluated': evaluated_clients,
            'samples_evaluated': len(all_predictions)
        }

    def _print_progress(self, round_num, total_rounds, start_time):
        elapsed = time.time() - start_time
        progress = (round_num + 1) / total_rounds * 100
        avg_time = elapsed / (round_num + 1)
        remaining_rounds = total_rounds - round_num - 1
        estimated_remaining = avg_time * remaining_rounds

        print(f"📊📊 进度: {round_num + 1}/{total_rounds}轮 ({progress:.1f}%) - "
              f"预计剩余: {estimated_remaining / 60:.1f}分钟")

    def _save_checkpoint(self, model, round_num):
        checkpoint_path = os.path.join(
            self.models_dir, f'checkpoint_round_{round_num}.pth'
        )
        torch.save(model.state_dict(), checkpoint_path)
        print(f"💾💾 检查点已保存: {checkpoint_path}")

    def _print_metrics(self, dataset_name, metrics):
        print(f"📊📊 {dataset_name}性能:")
        print(f"   绝对误差 - RMSE: {metrics['avg_rmse']:.6f}")
        print(f"   绝对误差 - MAE:  {metrics['avg_mae']:.6f}")
        if 'range_error_percent' in metrics:
            print(f"   相对误差 - 范围误差: {metrics['range_error_percent']:.2f}% (基于[0,1.6]范围)")
        if 'mape_percent' in metrics and metrics['mape_percent'] > 0:
            print(f"   相对误差 - MAPE: {metrics['mape_percent']:.2f}% (平均绝对百分比)")

    def _print_compute_statistics(self, compute_stats):
        print(f"⚡⚡ 计算开销统计:")
        print(f"   总客户端计算时间: {compute_stats['total_client_time']:.2f}s")
        print(f"   总服务器计算时间: {compute_stats['total_server_time']:.2f}s")
        print(f"   总数据传输时间: {compute_stats['total_transfer_time']:.2f}s")
        print(f"   平均客户端时间: {compute_stats['avg_client_time']:.2f}s")
        print(f"   平均服务器时间: {compute_stats['avg_server_time']:.2f}s")
        print(f"   客户端/服务器时间比: {compute_stats['client_device_ratio']:.2f}")
        print(f"   总客户端训练次数: {compute_stats['total_clients']}")
        print(f"   总训练轮次: {compute_stats['total_rounds']}")

    def _finalize_training(self, model, history, test_clients, initial_metrics,
                           start_time, total_rounds, train_clients, val_clients):
        total_time = time.time() - start_time

        print("\n🧪🧪 最终测试评估:")
        test_metrics = self.evaluate_model(model, test_clients, 'test')

        if test_metrics:
            self._print_metrics("测试集", test_metrics)

        final_model_path = os.path.join(self.models_dir, 'global_model_final.pth')
        torch.save(model.state_dict(), final_model_path)

        compute_stats = self.device_manager.get_compute_statistics()

        final_result = {
            'config': self.config,
            'total_rounds': total_rounds,
            'total_time': total_time,
            'training_history': history,
            'evaluation_metrics': {
                'initial': initial_metrics,
                'final': test_metrics
            },
            'client_statistics': {
                'train_clients': len(train_clients),
                'val_clients': len(val_clients),
                'test_clients': len(test_clients)
            },
            'compute_statistics': compute_stats,
            'device_configuration': {
                'server_device': str(self.device_manager.server_device),
                'client_device': str(self.device_manager.client_device),
                'gpu_name': self.device_manager.gpu_name
            },
            'final_model_path': final_model_path,
            'completion_time': datetime.now().isoformat()
        }

        self._save_final_result(final_result)
        self._print_final_summary(final_result, compute_stats)

        return final_result

    def _save_final_result(self, result):
        result_path = os.path.join(self.results_dir, 'final_training_result.json')

        def serialize(obj):
            if isinstance(obj, (np.integer, np.floating)):
                return float(obj)
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, dict):
                return {k: serialize(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [serialize(item) for item in obj]
            return obj

        with open(result_path, 'w', encoding='utf-8') as f:
            json.dump(serialize(result), f, ensure_ascii=False, indent=2)

        print(f"💾💾 训练结果已保存: {result_path}")

    def _print_final_summary(self, result, compute_stats):
        test_metrics = result['evaluation_metrics']['final']

        print("\n🎯🎯 训练完成总结:")
        print("=" * 60)
        if test_metrics:
            print(f"📊📊 测试集性能:")
            print(f"   绝对误差 - RMSE: {test_metrics['avg_rmse']:.6f}")
            print(f"   绝对误差 - MAE:  {test_metrics['avg_mae']:.6f}")
            if 'range_error_percent' in test_metrics:
                print(f"   相对误差 - 范围误差: {test_metrics['range_error_percent']:.2f}%")
            if 'mape_percent' in test_metrics and test_metrics['mape_percent'] > 0:
                print(f"   相对误差 - MAPE: {test_metrics['mape_percent']:.2f}%")

        print(f"⏱⏱⏱️  总耗时: {result['total_time']:.2f}秒")
        print(f"🔄🔄 总轮次: {result['total_rounds']}轮")
        print(f"💾💾 模型保存: {result['final_model_path']}")

        self._print_compute_statistics(compute_stats)
        print("=" * 60)

    def run_training(self):
        print("🚀🚀 开始FedAvg联邦学习训练（设备分离模式）")
        print("=" * 60)

        self._print_training_config()

        train_clients = self.get_all_client_ids('train')
        val_clients = self.get_all_client_ids('val')
        test_clients = self.get_all_client_ids('test')

        if not train_clients:
            raise ValueError("❌❌ 没有找到训练客户端数据")

        global_model = self.create_model()
        if global_model is None:
            return False

        print("\n🔍🔍 初始模型评估:")
        initial_metrics = self.evaluate_model(global_model, val_clients, 'val')
        if initial_metrics:
            self._print_metrics("初始验证集", initial_metrics)

        rounds = self.config.get('rounds', 100)
        clients_per_round = self.config.get('clients_per_round', 10)

        training_history = []
        start_time = time.time()

        for round_num in range(rounds):
            round_start = time.time()

            selected_clients = random.sample(
                train_clients,
                min(clients_per_round, len(train_clients))
            )

            print(f"\n🔁🔁 轮次 {round_num + 1}/{rounds} - 选择 {len(selected_clients)} 个客户端")

            client_updates = []
            for i, client_id in enumerate(selected_clients):
                update = self.client_local_train(global_model, client_id)
                if update:
                    client_updates.append(update)
                    if (i + 1) % 5 == 0:
                        print(f"   ✅ 已完成 {i + 1}/{len(selected_clients)} 个客户端")

            if client_updates:
                new_global_params = self.fedavg_aggregate(client_updates)
                global_model.load_state_dict(new_global_params)

                if (round_num % 10 == 0) or (round_num == rounds - 1):
                    val_metrics = self.evaluate_model(global_model, val_clients, 'val')
                    round_info = {
                        'round': round_num,
                        'clients_trained': len(client_updates),
                        'val_metrics': val_metrics,
                        'round_time': time.time() - round_start
                    }
                    training_history.append(round_info)

                    if val_metrics:
                        self._print_round_metrics(round_num, val_metrics)

                if (round_num % 50 == 0) or (round_num == rounds - 1):
                    self._save_checkpoint(global_model, round_num)

            if (round_num + 1) % 10 == 0:
                self._print_progress(round_num, rounds, start_time)

        final_result = self._finalize_training(
            global_model, training_history, test_clients, initial_metrics,
            start_time, rounds, train_clients, val_clients
        )

        return True

    def _print_training_config(self):
        print("📋📋 训练配置信息:")
        print(f"   项目根目录: {self.project_root}")
        print(f"   配置文件: {self.config_path}")
        print(f"   算法: FedAvg")
        print(f"   模型类型: {self.config.get('model_type', 'mlp')}")
        print(f"   训练轮次: {self.config.get('rounds', 100)}")
        print(f"   每轮客户端: {self.config.get('clients_per_round', 10)}")
        print(f"   本地轮次: {self.config.get('local_epochs', 5)}")
        print(f"   学习率: {self.config.get('learning_rate', 0.001)}")
        print(f"   输出目录: {self.output_dir}")
        print(f"   服务器设备: {self.device_manager.server_device}")
        print(f"   客户端设备: {self.device_manager.client_device}")

    def _print_round_metrics(self, round_num, metrics):
        print(f"📈📈 轮次 {round_num} 验证指标:")
        print(f"   绝对误差 - RMSE: {metrics['avg_rmse']:.6f}")
        print(f"   绝对误差 - MAE:  {metrics['avg_mae']:.6f}")
        if 'range_error_percent' in metrics:
            print(f"   相对误差 - 范围误差: {metrics['range_error_percent']:.2f}%")


def main():
    import argparse
    parser = argparse.ArgumentParser(description='FedAvg联邦学习训练脚本（设备分离版）')
    parser.add_argument('--config', type=str, help='配置文件路径（可选）')

    args = parser.parse_args()

    try:
        trainer = FedAvgTrainingSystem(args.config)
        success = trainer.run_training()
        return 0 if success else 1
    except Exception as e:
        print(f"❌❌ 训练失败: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    exit(main())
