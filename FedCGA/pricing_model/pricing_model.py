import torch
import torch.nn as nn
import numpy as np


def init_linear_layers(model):
    for module in model.modules():
        if isinstance(module, nn.Linear):
            nn.init.xavier_uniform_(module.weight)
            nn.init.constant_(module.bias, 0.1)


class MLPPricingModel(nn.Module):
    def __init__(self, input_dim=19, hidden_dim=32, output_dim=1, dropout=0.1):
        super().__init__()
        self.model_name = "MLP"
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim

        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, output_dim),
        )

        init_linear_layers(self)

    def forward(self, x):
        return self.network(x).squeeze()

    def predict_price(self, features, base_price=10.0):
        with torch.no_grad():
            if isinstance(features, np.ndarray):
                features = torch.FloatTensor(features)

            price_factor = self.forward(features).item()
            price_factor = max(0.5, min(price_factor, 2.0))
            return base_price * price_factor

    def get_model_info(self):
        return {
            "name": self.model_name,
            "input_dim": self.input_dim,
            "hidden_dim": self.hidden_dim,
            "output_dim": self.output_dim,
            "total_params": sum(p.numel() for p in self.parameters()),
        }


class ResidualPricingModel(nn.Module):
    def __init__(self, input_dim=19, hidden_dim=64, output_dim=1, dropout=0.1):
        super().__init__()
        self.model_name = "Residual"
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim

        self.layer1 = nn.Linear(input_dim, hidden_dim)
        self.activation = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
        self.layer2 = nn.Linear(hidden_dim, output_dim)
        self.residual = nn.Linear(input_dim, hidden_dim) if input_dim != hidden_dim else None

        init_linear_layers(self)

    def forward(self, x):
        h = self.dropout(self.activation(self.layer1(x)))

        if self.residual is not None:
            h = h + self.residual(x)

        return self.layer2(h).squeeze()

    def get_model_info(self):
        return {
            "name": self.model_name,
            "input_dim": self.input_dim,
            "hidden_dim": self.hidden_dim,
            "output_dim": self.output_dim,
            "total_params": sum(p.numel() for p in self.parameters()),
        }


class WidePricingModel(nn.Module):
    def __init__(self, input_dim=19, hidden_dim=128, output_dim=1, dropout=0.15):
        super().__init__()
        self.model_name = "Wide"
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim

        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, output_dim),
        )

        init_linear_layers(self)

    def forward(self, x):
        return self.network(x).squeeze()

    def get_model_info(self):
        return {
            "name": self.model_name,
            "input_dim": self.input_dim,
            "hidden_dim": self.hidden_dim,
            "output_dim": self.output_dim,
            "total_params": sum(p.numel() for p in self.parameters()),
        }


class TransformerPricingModel(nn.Module):
    def __init__(self, input_dim=19, d_model=128, nhead=8, num_layers=3, dropout=0.1):
        super().__init__()
        self.model_name = "Transformer"
        self.input_dim = input_dim
        self.d_model = d_model
        self.nhead = nhead
        self.num_layers = num_layers

        self.input_projection = nn.Linear(input_dim, d_model)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dropout=dropout,
            batch_first=True,
        )

        self.transformer = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_layers,
        )

        self.dropout = nn.Dropout(dropout)
        self.output_layer = nn.Linear(d_model, 1)

        init_linear_layers(self)

    def forward(self, x):
        x = self.input_projection(x)
        x = x.unsqueeze(1)
        x = self.transformer(x)
        x = x.squeeze(1)
        x = self.dropout(x)
        return self.output_layer(x).squeeze()

    def predict_price(self, features, base_price=10.0):
        with torch.no_grad():
            if isinstance(features, np.ndarray):
                features = torch.FloatTensor(features)

            price_factor = self.forward(features).item()
            price_factor = max(0.5, min(price_factor, 2.0))
            return base_price * price_factor

    def get_model_info(self):
        return {
            "name": self.model_name,
            "input_dim": self.input_dim,
            "d_model": self.d_model,
            "nhead": self.nhead,
            "num_layers": self.num_layers,
            "total_params": sum(p.numel() for p in self.parameters()),
        }


def create_pricing_model(model_name="mlp", **kwargs):
    model_registry = {
        "mlp": MLPPricingModel,
        "residual": ResidualPricingModel,
        "wide": WidePricingModel,
        "transformer": TransformerPricingModel,
    }

    if model_name not in model_registry:
        available = ", ".join(model_registry.keys())
        raise ValueError(f"Unknown model: {model_name}. Available models: {available}")

    return model_registry[model_name](**kwargs)


def get_model_info(model_name):
    info = {
        "mlp": {
            "name": "MLP",
            "type": "feed-forward",
            "complexity": "low",
        },
        "residual": {
            "name": "Residual",
            "type": "residual network",
            "complexity": "medium",
        },
        "wide": {
            "name": "Wide",
            "type": "wide feed-forward network",
            "complexity": "medium",
        },
        "transformer": {
            "name": "Transformer",
            "type": "transformer encoder",
            "complexity": "high",
        },
    }

    return info.get(model_name, {})