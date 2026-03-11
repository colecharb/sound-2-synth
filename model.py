"""
model.py — MLP Parameter Predictor

Maps OpenL3 embeddings (512-dim) to FM synth parameters (10-dim).
Architecture: 512 → 256 → 128 → 10 with ReLU activations.
"""

import torch
import torch.nn as nn
import math


# Parameter ranges for output scaling (from PARAMETER_ORDER.md)
PARAM_RANGES = {
    "carrier_freq": (20.0, 2000.0),
    "mod_ratio": (0.5, 4.0),
    "mod_index": (0.0, 20.0),
    "attack": (0.001, 1.0),
    "decay": (0.001, 2.0),
    "sustain": (0.0, 1.0),
    "release": (0.001, 2.0),
    "note_duration": (0.05, 2.0),
    "lowpass_freq": (20.0, 20000.0),
    "highpass_freq": (20.0, 10000.0),
}

PARAM_NAMES = [
    "carrier_freq",
    "mod_ratio",
    "mod_index",
    "attack",
    "decay",
    "sustain",
    "release",
    "note_duration",
    "lowpass_freq",
    "highpass_freq",
]

# Which parameters use log-scale (perceptually uniform)
LOG_SCALE_PARAMS = {"lowpass_freq", "highpass_freq"}


class ParameterPredictor(nn.Module):
    """MLP that maps OpenL3 embeddings to FM synth parameters.
    
    Input: 512-dimensional OpenL3 embedding
    Output: 10-dimensional parameter vector (clamped to valid ranges)
    
    The output uses sigmoid activation and is scaled to each parameter's
    valid range. For log-scale parameters, scaling is done in log space.
    """
    
    def __init__(self, embedding_dim: int = 512, dropout: float = 0.1):
        super().__init__()
        
        # MLP layers: 512 → 256 → 128 → 10
        self.fc1 = nn.Linear(embedding_dim, 256)
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(dropout)
        
        self.fc2 = nn.Linear(256, 128)
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(dropout)
        
        self.fc3 = nn.Linear(128, len(PARAM_NAMES))
        self.sigmoid = nn.Sigmoid()
        
        # Pre-compute scaling factors for output normalization
        self._register_scaling_buffers()
    
    def _register_scaling_buffers(self):
        """Register min/max bounds for each parameter as buffers."""
        param_mins = []
        param_maxs = []
        param_is_log = []
        
        for param_name in PARAM_NAMES:
            min_val, max_val = PARAM_RANGES[param_name]
            param_mins.append(min_val)
            param_maxs.append(max_val)
            param_is_log.append(param_name in LOG_SCALE_PARAMS)
        
        self.register_buffer("param_mins", torch.tensor(param_mins, dtype=torch.float32))
        self.register_buffer("param_maxs", torch.tensor(param_maxs, dtype=torch.float32))
        self.register_buffer("param_is_log", torch.tensor(param_is_log, dtype=torch.bool))
    
    def forward(self, embedding: torch.Tensor) -> torch.Tensor:
        """
        Predict parameters from embedding.
        
        Parameters
        ----------
        embedding : Tensor
            Shape (batch_size, 512) or (512,) OpenL3 embedding(s)
        
        Returns
        -------
        Tensor
            Shape (batch_size, 10) or (10,) parameter vector(s), clamped to valid ranges
        """
        # Handle both batched and single embeddings
        squeeze_output = embedding.ndim == 1
        if squeeze_output:
            embedding = embedding.unsqueeze(0)
        
        # Forward pass
        x = self.relu1(self.fc1(embedding))
        x = self.dropout1(x)
        x = self.relu2(self.fc2(x))
        x = self.dropout2(x)
        x = self.fc3(x)  # Raw logits
        x = self.sigmoid(x)  # Sigmoid to [0, 1]
        
        # Scale to parameter ranges
        params = self._scale_outputs(x)
        
        if squeeze_output:
            params = params.squeeze(0)
        
        return params
    
    def _scale_outputs(self, normalized: torch.Tensor) -> torch.Tensor:
        """Scale sigmoid outputs [0, 1] to parameter ranges.
        
        For linear-scale parameters: y = min + normalized * (max - min)
        For log-scale parameters:   y = exp(log(min) + normalized * (log(max) - log(min)))
        
        Parameters
        ----------
        normalized : Tensor
            Shape (batch_size, 10) with values in [0, 1]
        
        Returns
        -------
        Tensor
            Shape (batch_size, 10) with values in valid parameter ranges
        """
        batch_size = normalized.shape[0]
        params = torch.zeros_like(normalized)
        
        for i, param_name in enumerate(PARAM_NAMES):
            min_val = self.param_mins[i]
            max_val = self.param_maxs[i]
            norm_val = normalized[:, i]
            
            if param_name in LOG_SCALE_PARAMS:
                # Log-scale: interpolate in log space
                log_min = math.log(min_val)
                log_max = math.log(max_val)
                log_val = log_min + norm_val * (log_max - log_min)
                params[:, i] = torch.exp(log_val)
            else:
                # Linear scale: simple interpolation
                params[:, i] = min_val + norm_val * (max_val - min_val)
        
        return params
    
    def parameter_names(self) -> list[str]:
        """Return the ordered list of parameter names."""
        return PARAM_NAMES.copy()


if __name__ == "__main__":
    # Quick test
    model = ParameterPredictor()
    print(f"ParameterPredictor architecture:")
    print(model)
    print(f"\nTotal parameters: {sum(p.numel() for p in model.parameters())}")
    
    # Test forward pass
    dummy_embedding = torch.randn(1, 512)
    output = model(dummy_embedding)
    print(f"\nTest output shape: {output.shape}")
    print(f"Test output (first 5 params): {output[0, :5]}")
    print(f"Parameter ranges:")
    for i, name in enumerate(PARAM_NAMES[:5]):
        min_val, max_val = PARAM_RANGES[name]
        print(f"  {name:20} : [{min_val:8.2f}, {max_val:8.2f}]")
