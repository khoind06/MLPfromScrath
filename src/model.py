import numpy as np
from typing import List, Dict

class MLP:
    """
    MLP Optimized: ReLU + Balanced Class Weight + Batch Norm
    """
    
    def __init__(self, input_dim: int, hidden_dims: List[int], 
                 proj_dim: int = 64, dropout: float = 0.5,
                 weight_decay: float = 5e-4, use_batch_norm: bool = True):
        
        self.input_dim = input_dim
        self.hidden_dims = hidden_dims
        self.proj_dim = proj_dim
        self.dropout_rate = dropout
        self.weight_decay = weight_decay
        self.use_batch_norm = use_batch_norm
        
        self.params = {}
        self.grads = {}
        self.adam_m = {} 
        self.adam_v = {} 
        self.t = 0
        self.training = True
        
        # Running stats for Batch Norm
        self.bn_running_mean = {}
        self.bn_running_var = {}
        self.bn_momentum = 0.9
        
        # --- Initialization ---
        # 1. Projection Layer
        self.params['W_proj'] = np.random.randn(input_dim, proj_dim) * np.sqrt(2. / input_dim)
        self.params['b_proj'] = np.zeros(proj_dim)
        
        # 2. Hidden Layers
        prev_dim = proj_dim
        for i, dim in enumerate(hidden_dims):
            self.params[f'W{i+1}'] = np.random.randn(prev_dim, dim) * np.sqrt(2. / prev_dim)
            self.params[f'b{i+1}'] = np.zeros(dim)
            if self.use_batch_norm:
                self.params[f'gamma{i+1}'] = np.ones(dim)
                self.params[f'beta{i+1}'] = np.zeros(dim)
            prev_dim = dim
            
        # 3. Output Layer
        self.params['W_out'] = np.random.randn(prev_dim, 1) * np.sqrt(2. / (prev_dim + 1))
        self.params['b_out'] = np.zeros(1)
        
        # Init Adam states
        for k, v in self.params.items():
            self.adam_m[k] = np.zeros_like(v)
            self.adam_v[k] = np.zeros_like(v)

    def train(self): self.training = True
    def eval(self): self.training = False

    def forward(self, X_sparse):
        self.cache = {}
        
        # 1. Projection
        Z_proj = X_sparse @ self.params['W_proj'] + self.params['b_proj']
        A = np.maximum(0, Z_proj)
        self.cache['A_proj'] = A
        self.cache['X_sparse'] = X_sparse
        
        # 2. Hidden Layers
        num_layers = len(self.hidden_dims)
        for i in range(num_layers):
            W = self.params[f'W{i+1}']
            b = self.params[f'b{i+1}']
            Z = A @ W + b
            
            if self.use_batch_norm:
                Z = self._batch_norm_forward(Z, f'{i+1}')
            
            A = np.maximum(0, Z)
            
            if self.training:
                mask = (np.random.rand(*A.shape) > self.dropout_rate) / (1.0 - self.dropout_rate)
                A *= mask
                self.cache[f'mask{i+1}'] = mask
            
            self.cache[f'A{i+1}'] = A
            self.cache[f'Z{i+1}'] = Z
            self.cache[f'prev_A{i+1}'] = self.cache.get(f'A{i}', self.cache['A_proj']) if i > 0 else self.cache['A_proj']

        # 3. Output
        W_out = self.params['W_out']
        b_out = self.params['b_out']
        logits = A @ W_out + b_out
        self.cache['final_A'] = A
        
        return 1 / (1 + np.exp(-np.clip(logits, -15, 15)))

    def predict(self, X):
        self.eval()
        return self.forward(X).flatten()

    def _batch_norm_forward(self, x, name):
        if self.training:
            mean = np.mean(x, axis=0)
            var = np.var(x, axis=0)
            
            # Update running stats
            if name not in self.bn_running_mean:
                self.bn_running_mean[name] = mean
                self.bn_running_var[name] = var
            else:
                self.bn_running_mean[name] = self.bn_momentum * self.bn_running_mean[name] + (1 - self.bn_momentum) * mean
                self.bn_running_var[name] = self.bn_momentum * self.bn_running_var[name] + (1 - self.bn_momentum) * var
            
            x_norm = (x - mean) / np.sqrt(var + 1e-8)
            y = self.params[f'gamma{name}'] * x_norm + self.params[f'beta{name}']
            self.cache[f'bn_mean_{name}'] = mean
            self.cache[f'bn_var_{name}'] = var
            self.cache[f'bn_x_norm_{name}'] = x_norm
        else:
            mean = self.bn_running_mean.get(name, np.zeros(x.shape[1]))
            var = self.bn_running_var.get(name, np.ones(x.shape[1]))
            x_norm = (x - mean) / np.sqrt(var + 1e-8)
            y = self.params.get(f'gamma{name}', np.ones(x.shape[1])) * x_norm + self.params.get(f'beta{name}', np.zeros(x.shape[1]))
        
        return y

    def backward(self, y_pred, y_true, lr, clip_norm):
        m = len(y_true)
        y_pred = y_pred.reshape(-1, 1)
        y_true = y_true.reshape(-1, 1)
        
        # === BALANCED CLASS WEIGHT ===
        pos_ratio = np.mean(y_true)
        neg_ratio = 1 - pos_ratio
        
        # Milder balancing: (neg/pos)^0.25, clipped 1-5
        pos_weight = (neg_ratio / (pos_ratio + 1e-8)) ** 0.25
        pos_weight = np.clip(pos_weight, 1.0, 5.0)
        
        weights = np.where(y_true == 1, pos_weight, 1.0)
        
        # dLoss/dLogits
        dLogits = weights * (y_pred - y_true) / m
        
        # Output Layer
        dA = dLogits @ self.params['W_out'].T
        self.grads['W_out'] = self.cache['final_A'].T @ dLogits
        self.grads['b_out'] = np.sum(dLogits, axis=0)
        
        # Hidden Layers
        num_layers = len(self.hidden_dims)
        for i in range(num_layers - 1, -1, -1):
            if self.training:
                dA *= self.cache[f'mask{i+1}']
            
            dZ = dA * (self.cache[f'Z{i+1}'] > 0)
            
            if self.use_batch_norm:
                dZ = self._batch_norm_backward(dZ, f'{i+1}')
            
            prev_A = self.cache[f'prev_A{i+1}']
            self.grads[f'W{i+1}'] = prev_A.T @ dZ
            self.grads[f'b{i+1}'] = np.sum(dZ, axis=0)
            
            if i > 0:
                dA = dZ @ self.params[f'W{i+1}'].T
            else:
                dA_proj = dZ @ self.params['W1'].T
        
        # Projection Layer
        dZ_proj = dA_proj * (self.cache['A_proj'] > 0)
        self.grads['W_proj'] = self.cache['X_sparse'].T @ dZ_proj
        self.grads['b_proj'] = np.sum(dZ_proj, axis=0)

        # Adam Update
        self._apply_adam(lr, clip_norm)
        
        # Loss hiển thị
        epsilon = 1e-15
        y_pred = np.clip(y_pred, epsilon, 1 - epsilon)
        loss = -np.mean(weights * (y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred)))
        return loss

    def _batch_norm_backward(self, dout, name):
        x_norm = self.cache[f'bn_x_norm_{name}']
        var = self.cache[f'bn_var_{name}']
        gamma = self.params[f'gamma{name}']
        N = dout.shape[0]
        
        dgamma = np.sum(dout * x_norm, axis=0)
        dbeta = np.sum(dout, axis=0)
        self.grads[f'gamma{name}'] = dgamma
        self.grads[f'beta{name}'] = dbeta
        
        dx_norm = dout * gamma
        dx = (1. / N) / np.sqrt(var + 1e-8) * (N * dx_norm - np.sum(dx_norm, axis=0) - x_norm * np.sum(dx_norm * x_norm, axis=0))
        return dx

    def _apply_adam(self, lr, clip_norm):
        self.t += 1
        beta1, beta2 = 0.9, 0.999
        eps = 1e-8
        
        # Clip Gradient Global Norm
        total_norm = 0
        for k in self.grads:
            total_norm += np.sum(self.grads[k]**2)
        total_norm = np.sqrt(total_norm)
        scale = clip_norm / (total_norm + 1e-6) if total_norm > clip_norm else 1.0
        
        for k in self.params:
            grad = self.grads[k] * scale
            
            # Weight Decay (L2)
            if 'W' in k: grad += self.weight_decay * self.params[k]
            
            self.adam_m[k] = beta1 * self.adam_m[k] + (1 - beta1) * grad
            self.adam_v[k] = beta2 * self.adam_v[k] + (1 - beta2) * (grad**2)
            
            m_hat = self.adam_m[k] / (1 - beta1**self.t)
            v_hat = self.adam_v[k] / (1 - beta2**self.t)
            
            self.params[k] -= lr * m_hat / (np.sqrt(v_hat) + eps)