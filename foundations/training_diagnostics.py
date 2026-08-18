import torch
import torch.nn as nn
from typing import List, Dict


class Solution:

    def compute_activation_stats(self, model: nn.Module, x: torch.Tensor) -> List[Dict[str, float]]:
        stats = []
        hooks = []

        def make_hook(storage):
            def hook(module, input, output):
                storage.append(output.detach())
            return hook
        outputs_per_layer = []
        linear_layers = [m for m in model.modules() if isinstance(m, nn.Linear)]

        for layer in linear_layers:
            storage = []
            h = layer.register_forward_hook(make_hook(storage))
            hooks.append(h)
            outputs_per_layer.append(storage)
        with torch.no_grad():
            model(x)
        for h in hooks:
            h.remove()
        for storage in outputs_per_layer:
            out = storage[0]
            mean = out.mean().item()
            std = out.std().item()
            dead_mask = (out <= 0).all(dim=0)
            dead_fraction = dead_mask.float().mean().item()
            stats.append({
                'mean': round(mean, 4),
                'std': round(std, 4),
                'dead_fraction':
                round(dead_fraction, 4)
            })          
        return stats    
        # Forward pass through model layer by layer
        # After each nn.Linear, record: mean, std, dead_fraction
        # Run with torch.no_grad(). Round to 4 decimals.
        pass

    def compute_gradient_stats(self, model: nn.Module, x: torch.Tensor, y: torch.Tensor) -> List[Dict[str, float]]:
        model.zero_grad()

        criterion = nn.MSELoss()
        output = model(x)
        loss = criterion(output, y)
        loss.backward()

        state = []
        for layer in model.modules():
            if isinstance(layer, nn.Linear):
                grad = layer.weight.grad 
                mean = grad.mean().item()
                std = grad.std().item()
                norm = torch.norm(grad).item()

                state.append({
                    'mean': round(mean, 4),
                    'std': round(std, 4),
                    'norm': round(norm,4)
                })
        return state


        # Forward + backward pass with nn.MSELoss
        # For each nn.Linear layer's weight gradient, record: mean, std, norm
        # Call model.zero_grad() first. Round to 4 decimals.
        pass

    def diagnose(self, activation_stats: List[Dict[str, float]], gradient_stats: List[Dict[str, float]]) -> str:
        if any(layer['dead_fraction'] > 0.5 for layer in activation_stats):
            return 'dead_neurons'

        if any(layer['norm'] > 1000 for layer in gradient_stats):
            return 'exploding_gradients'
        if gradient_stats[-1]['norm'] < 1e-5:
            return 'vanishing_gradients'
        if any(layer['std'] < 0.1 for layer in activation_stats):
            return 'vanishing_gradients'
        if any(layer['std'] > 10.0 for layer in activation_stats):
            return 'exploding_gradients'


        return 'healthy'
        # Classify network health based on the stats
        # Return: 'dead_neurons', 'exploding_gradients', 'vanishing_gradients', or 'healthy'
        # Check in priority order (see problem description for thresholds)
        pass
