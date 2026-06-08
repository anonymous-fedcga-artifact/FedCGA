#!/usr/bin/env python3
import os
import sys
import json
import time
import yaml
import random
import argparse
from datetime import datetime

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.normpath(os.path.join(current_dir, ".."))
sys.path.insert(0, project_root)

from pricing_model.pricing_model import create_pricing_model


class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, torch.Tensor):
            return obj.item() if obj.numel() == 1 else obj.cpu().numpy().tolist()
        return super().default(obj)


class ContextFeatureExtractor:
    def extract_client_context(self, client_id, X, y):
        if isinstance(X, torch.Tensor):
            X = X.detach().cpu().numpy()
        if isinstance(y, torch.Tensor):
            y = y.detach().cpu().numpy()

        context = {
            "sample_count": int(X.shape[0]),
            "feature_dim": int(X.shape[1]),
            "label_mean": float(np.mean(y)),
            "label_std": float(np.std(y)),
            "label_min": float(np.min(y)),
            "label_max": float(np.max(y)),
            "previous_performance": 0.5,
        }

        if X.shape[0] > 1 and X.shape[1] > 1:
            corr = np.corrcoef(X.T)
            context["feature_correlation"] = float(np.nan_to_num(corr).mean())
        else:
            context["feature_correlation"] = 0.0

        return context

    def choose_strategy(self, context):
        sample_count = context.get("sample_count", 0)
        label_std = context.get("label_std", 1.0)

        if sample_count < 50 or label_std > 2.0:
            return "conservative"
        if sample_count > 200 and label_std < 0.5:
            return "aggressive"
        return "adaptive"


class ContextAwareALA:
    def __init__(
        self,
        client_id,
        loss_fn,
        train_data,
        batch_size=32,
        rand_percent=80,
        layer_idx=1,
        eta=1.0,
        threshold=0.1,
        num_pre_loss=10,
        context_extractor=None,
    ):
        self.client_id = client_id
        self.loss_fn = loss_fn
        self.train_data = train_data
        self.batch_size = batch_size
        self.rand_percent = rand_percent
        self.layer_idx = layer_idx
        self.eta = eta
        self.threshold = threshold
        self.num_pre_loss = num_pre_loss
        self.context_extractor = context_extractor or ContextFeatureExtractor()
        self.weights = None
        self.start_phase = True

    def _extract_context(self):
        X_list, y_list = [], []

        for x, y in self.train_data:
            X_list.append(x)
            y_list.append(y)

        X = torch.stack(X_list, dim=0)
        y = torch.stack(y_list, dim=0).squeeze()

        return self.context_extractor.extract_client_context(self.client_id, X, y)

    def _base_weight(self, context, strategy):
        sample_count = context.get("sample_count", 100)
        label_std = context.get("label_std", 1.0)
        previous_performance = context.get("previous_performance", 0.5)

        data_quality = min(1.0, np.log1p(float(sample_count)) / 10.0)
        stability = 1.0 / (1.0 + float(label_std))

        if strategy == "conservative":
            weight = 0.2 + 0.3 * data_quality
        elif strategy == "aggressive":
            weight = 0.6 + 0.4 * data_quality
        else:
            weight = 0.3 + 0.4 * data_quality + 0.3 * previous_performance

        return float(np.clip(weight * stability, 0.0, 1.0))

    def _init_weights(self, params, context, strategy):
        base = self._base_weight(context, strategy)
        weights = []

        for idx, param in enumerate(params):
            factor = 1.0
            if idx == 0 and len(param.shape) == 2:
                factor = 0.8
            elif len(param.shape) == 1:
                factor = 1.2

            value = float(np.clip(base * factor, 0.0, 1.0))
            weights.append(torch.full_like(param.data, value))

        return weights

    def _sample_loader(self):
        dataset_size = len(self.train_data)
        sample_size = max(1, int(dataset_size * self.rand_percent / 100))
        sample_size = min(sample_size, dataset_size)

        start = random.randint(0, max(0, dataset_size - sample_size))
        indices = list(range(start, start + sample_size))

        subset = torch.utils.data.Subset(self.train_data, indices)
        return torch.utils.data.DataLoader(
            subset,
            batch_size=self.batch_size,
            shuffle=True,
            drop_last=False,
        )

    def adaptive_local_aggregation(self, global_model, local_model):
        context = self._extract_context()
        strategy = self.context_extractor.choose_strategy(context)

        global_model = global_model.cpu()
        local_model = local_model.cpu()

        params_g = list(global_model.parameters())
        params_l = list(local_model.parameters())

        if self.layer_idx > 0:
            for p_l, p_g in zip(params_l[:-self.layer_idx], params_g[:-self.layer_idx]):
                p_l.data = p_g.data.clone()

        params_l_tail = params_l[-self.layer_idx:] if self.layer_idx > 0 else params_l
        params_g_tail = params_g[-self.layer_idx:] if self.layer_idx > 0 else params_g

        if self.weights is None:
            self.weights = self._init_weights(params_l_tail, context, strategy)

        loader = self._sample_loader()
        optimizer = torch.optim.SGD(params_l_tail, lr=0.0)

        losses = []
        best_loss = float("inf")
        patience = 0

        for _ in range(self.num_pre_loss):
            epoch_loss = 0.0
            batch_count = 0

            for x, y in loader:
                optimizer.zero_grad()

                for p_l, p_g, w in zip(params_l_tail, params_g_tail, self.weights):
                    p_l.data = p_l.data + (p_g.data - p_l.data) * w

                output = local_model(x).squeeze()
                loss = self.loss_fn(output, y)
                loss.backward()

                for p_l, p_g, w in zip(params_l_tail, params_g_tail, self.weights):
                    if p_l.grad is None:
                        continue

                    grad = p_l.grad * (p_g.data - p_l.data)
                    step = self.eta * 0.5 if grad.norm().item() > 1.0 else self.eta
                    w.data = torch.clamp(w.data - step * grad, 0.0, 1.0)

                epoch_loss += loss.item()
                batch_count += 1

            if batch_count == 0:
                break

            avg_loss = epoch_loss / batch_count
            losses.append(avg_loss)

            if avg_loss < best_loss - 1e-4:
                best_loss = avg_loss
                patience = 0
            else:
                patience += 1

            if patience >= 3:
                break

            if len(losses) >= self.num_pre_loss and np.std(losses[-self.num_pre_loss:]) < self.threshold:
                break

        self.start_phase = False
        return float(losses[-1]) if losses else 0.0, strategy


class FedCGATrainingSystem:
    def __init__(self, config_path="configs/context_aware_fedala.yaml"):
        self.script_dir = os.path.dirname(os.path.abspath(__file__))
        self.project_root = os.path.normpath(os.path.join(self.script_dir, ".."))

        if not os.path.isabs(config_path):
            config_path = os.path.join(self.project_root, config_path)

        self.config_path = os.path.normpath(config_path)
        self.config = self._load_config()

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.context_extractor = ContextFeatureExtractor()
        self.client_contexts = {}

        self._setup_output_dirs()

    def _load_config(self):
        if not os.path.exists(self.config_path):
            raise FileNotFoundError(f"Config file not found: {self.config_path}")

        with open(self.config_path, "r", encoding="utf-8") as f:
            config = yaml.safe_load(f)

        if "fedcga" in config:
            return config["fedcga"]
        if "fedala" in config:
            return config["fedala"]
        return config

    def _setup_output_dirs(self):
        model_type = self.config.get("model_type", "mlp")
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        self.output_dir = os.path.join(
            self.project_root,
            "outputs",
            f"fedcga_{model_type}_{timestamp}",
        )
        self.models_dir = os.path.join(self.output_dir, "global_models")
        self.results_dir = os.path.join(self.output_dir, "training_results")

        os.makedirs(self.models_dir, exist_ok=True)
        os.makedirs(self.results_dir, exist_ok=True)

    def get_all_client_ids(self, data_type="train"):
        features_dir = os.path.join(
            self.project_root,
            "data",
            "processed",
            f"{data_type}_features",
            "features",
        )

        if not os.path.exists(features_dir):
            return []

        files = [
            f for f in os.listdir(features_dir)
            if f.startswith("driver_") and f.endswith(".npz")
        ]
        return [f.replace("driver_", "").replace(".npz", "") for f in files]

    def load_client_data(self, client_id, data_type="train"):
        path = os.path.join(
            self.project_root,
            "data",
            "processed",
            f"{data_type}_features",
            "features",
            f"driver_{client_id}.npz",
        )

        if not os.path.exists(path):
            return None, None

        data = np.load(path)
        X = torch.FloatTensor(data["X"])
        y = torch.FloatTensor(data["y"]).squeeze()

        if client_id not in self.client_contexts:
            self.client_contexts[client_id] = self.context_extractor.extract_client_context(
                client_id,
                X,
                y,
            )

        return X, y

    def create_model(self):
        model_type = self.config.get("model_type", "mlp")
        model_params = self.config.get("model_params", {}).get(model_type, {})
        return create_pricing_model(model_type, **model_params).to(self.device)

    def client_local_train(self, global_model, client_id):
        X, y = self.load_client_data(client_id, "train")
        if X is None:
            return None

        local_epochs = self.config.get("local_epochs", 5)
        learning_rate = self.config.get("learning_rate", 0.001)
        ala_config = self.config.get("ala_config", {})

        dataset = torch.utils.data.TensorDataset(X, y)

        local_model = self.create_model().cpu()
        global_state = {k: v.detach().cpu() for k, v in global_model.state_dict().items()}
        local_model.load_state_dict(global_state)

        ala = ContextAwareALA(
            client_id=client_id,
            loss_fn=nn.MSELoss(),
            train_data=dataset,
            batch_size=ala_config.get("batch_size", 32),
            rand_percent=ala_config.get("rand_percent", 80),
            layer_idx=ala_config.get("layer_idx", 1),
            eta=ala_config.get("eta", 1.0),
            threshold=ala_config.get("threshold", 0.1),
            num_pre_loss=ala_config.get("num_pre_loss", 10),
            context_extractor=self.context_extractor,
        )

        ala_loss, strategy = ala.adaptive_local_aggregation(global_model, local_model)

        local_model.train()
        optimizer = torch.optim.Adam(local_model.parameters(), lr=learning_rate)
        criterion = nn.MSELoss()
        loader = torch.utils.data.DataLoader(dataset, batch_size=32, shuffle=True)

        losses = []
        for _ in range(local_epochs):
            epoch_loss = 0.0
            for batch_X, batch_y in loader:
                optimizer.zero_grad()
                pred = local_model(batch_X).squeeze()
                loss = criterion(pred, batch_y)
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item()

            losses.append(epoch_loss / max(1, len(loader)))

        return {
            "model_params": local_model.state_dict(),
            "num_samples": int(len(X)),
            "client_id": client_id,
            "train_loss": float(losses[-1] if losses else 0.0),
            "ala_loss": float(ala_loss),
            "strategy": strategy,
        }

    def federated_aggregate(self, client_updates):
        if not client_updates:
            raise ValueError("No client updates to aggregate.")

        total_samples = sum(update["num_samples"] for update in client_updates)
        averaged_params = {}

        first_params = client_updates[0]["model_params"]

        for name in first_params:
            weighted_sum = torch.zeros_like(first_params[name], device=self.device)

            for update in client_updates:
                weight = update["num_samples"] / total_samples
                param = update["model_params"][name].to(self.device)
                weighted_sum += param * weight

            averaged_params[name] = weighted_sum

        return averaged_params

    def evaluate_model(self, model, client_ids, data_type="val"):
        if not client_ids:
            return None

        model = model.cpu()
        model.eval()

        predictions = []
        targets = []
        evaluated_clients = 0

        with torch.no_grad():
            for client_id in client_ids:
                X, y = self.load_client_data(client_id, data_type)
                if X is None:
                    continue

                pred = model(X).squeeze().numpy()
                predictions.extend(np.atleast_1d(pred))
                targets.extend(y.numpy())
                evaluated_clients += 1

        model = model.to(self.device)

        if not predictions:
            return None

        rmse = float(np.sqrt(mean_squared_error(targets, predictions)))
        mae = float(mean_absolute_error(targets, predictions))
        r2 = float(r2_score(targets, predictions))

        return {
            "avg_rmse": rmse,
            "avg_mae": mae,
            "r2_score": r2,
            "clients_evaluated": int(evaluated_clients),
            "samples_evaluated": int(len(predictions)),
        }

    def save_result(self, result):
        path = os.path.join(self.results_dir, "training_result.json")
        with open(path, "w", encoding="utf-8") as f:
            json.dump(result, f, indent=2, cls=NumpyEncoder)
        return path

    def run_training(self):
        train_clients = self.get_all_client_ids("train")
        val_clients = self.get_all_client_ids("val")
        test_clients = self.get_all_client_ids("test")

        if not train_clients:
            raise RuntimeError("No training clients found.")

        global_model = self.create_model()
        initial_metrics = self.evaluate_model(global_model, val_clients, "val")

        rounds = self.config.get("rounds", 30)
        clients_per_round = self.config.get("clients_per_round", 50)

        history = []
        start_time = time.time()

        for round_idx in range(rounds):
            selected_clients = random.sample(
                train_clients,
                min(clients_per_round, len(train_clients)),
            )

            client_updates = []
            strategy_counts = {
                "conservative": 0,
                "adaptive": 0,
                "aggressive": 0,
            }

            for client_id in selected_clients:
                update = self.client_local_train(global_model, client_id)
                if update is None:
                    continue

                client_updates.append(update)
                strategy_counts[update["strategy"]] += 1

            if not client_updates:
                continue

            new_params = self.federated_aggregate(client_updates)
            global_model.load_state_dict(new_params)

            if round_idx == 0 or (round_idx + 1) % 10 == 0 or round_idx == rounds - 1:
                val_metrics = self.evaluate_model(global_model, val_clients, "val")
                history.append(
                    {
                        "round": round_idx + 1,
                        "clients_trained": len(client_updates),
                        "strategy_counts": strategy_counts,
                        "val_metrics": val_metrics,
                    }
                )

                if val_metrics:
                    print(
                        f"Round {round_idx + 1}/{rounds}: "
                        f"RMSE={val_metrics['avg_rmse']:.6f}, "
                        f"MAE={val_metrics['avg_mae']:.6f}, "
                        f"R2={val_metrics['r2_score']:.6f}"
                    )

        total_time = time.time() - start_time
        test_metrics = self.evaluate_model(global_model, test_clients, "test")

        model_path = os.path.join(self.models_dir, "final_model.pth")
        torch.save(global_model.state_dict(), model_path)

        result = {
            "algorithm": "FedCGA",
            "config": self.config,
            "rounds": rounds,
            "total_time": float(total_time),
            "client_statistics": {
                "train_clients": len(train_clients),
                "val_clients": len(val_clients),
                "test_clients": len(test_clients),
            },
            "evaluation_metrics": {
                "initial_validation": initial_metrics,
                "final_test": test_metrics,
            },
            "training_history": history,
            "final_model_path": model_path,
            "completion_time": datetime.now().isoformat(),
        }

        result_path = self.save_result(result)

        if test_metrics:
            print(
                f"Final Test: RMSE={test_metrics['avg_rmse']:.6f}, "
                f"MAE={test_metrics['avg_mae']:.6f}, "
                f"R2={test_metrics['r2_score']:.6f}"
            )

        print(f"Saved result to {result_path}")
        print(f"Saved model to {model_path}")

        return True


def main():
    parser = argparse.ArgumentParser(description="FedCGA training")
    parser.add_argument(
        "--config",
        type=str,
        default="configs/context_aware_fedala.yaml",
    )
    args = parser.parse_args()

    trainer = FedCGATrainingSystem(args.config)
    return 0 if trainer.run_training() else 1


if __name__ == "__main__":
    raise SystemExit(main())