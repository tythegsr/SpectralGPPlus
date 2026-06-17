# Why LBFGS Fails But Adam Works: Detailed Analysis

## Executive Summary

LBFGS fails because it performs **multiple forward passes per optimization step** (line search), which repeatedly evaluates the GP covariance matrix. This exposes numerical instabilities that Adam avoids by doing only **one forward/backward pass per step**.

## Key Differences

### 1. **Multiple Forward Passes (LBFGS) vs Single Forward Pass (Adam)**

#### LBFGS Behavior:
- **Line Search Algorithm**: LBFGS is a quasi-Newton method that performs line search
- **Closure Function**: The closure is called **multiple times per `optimizer.step()`** (up to `max_iter=20` times)
- **Each Closure Call**:
  1. Updates parameters (tries different step sizes)
  2. Re-evaluates the model: `output = model(model.train_inputs[0])`
  3. Computes loss: `loss = -mll(output, model.train_targets)`
  4. Performs Cholesky decomposition of covariance matrix

#### Adam Behavior:
- **Single Step**: Adam does **one forward/backward pass per step**
- **No Line Search**: Uses adaptive learning rates based on gradient history
- **Single Evaluation**: Only evaluates the model once per epoch

### 2. **Numerical Instability Cascade**

#### The Problem with LBFGS:
```
Step 1: Closure call #1 → Valid parameters → Valid covariance matrix → Valid loss
Step 2: LBFGS updates parameters (line search)
Step 3: Closure call #2 → Parameters slightly changed → Covariance matrix becomes ill-conditioned
Step 4: Cholesky decomposition fails → NaN in loss
Step 5: Closure returns penalty (1e6)
Step 6: LBFGS tries another step size → More invalid parameters
Step 7: Eventually all closure calls fail → Model state becomes invalid
```

#### Why Adam Avoids This:
```
Step 1: Forward pass → Valid parameters → Valid covariance matrix → Valid loss
Step 2: Backward pass → Compute gradients
Step 3: Update parameters (adaptive step size, bounded by learning rate)
Step 4: Next epoch: Start fresh with new parameters
```

### 3. **Parameter Update Behavior**

#### LBFGS Line Search:
- **Tries multiple step sizes** during line search
- **No bounds checking** on parameter values during line search
- Can push parameters into **invalid regions** (e.g., negative lengthscales after transformation, extremely small values)
- Even if gradients are valid, the **intermediate parameter values** during line search can be invalid

#### Adam Adaptive Updates:
- **Bounded step sizes** by learning rate and adaptive momentum
- **More conservative updates** (especially with `lr=0.1` and momentum)
- **Gradient clipping** (if NaN gradients are zeroed) prevents large jumps
- Parameters change **gradually** between epochs

### 4. **Covariance Matrix Conditioning**

#### The Critical Issue:
GP covariance matrices become **ill-conditioned** when:
- Lengthscales become very small → Kernel values become very large
- Lengthscales become very large → Kernel values become very small (numerical precision issues)
- Outputscale becomes extreme → Scales the entire covariance matrix
- Noise becomes very small → Matrix becomes nearly singular

#### LBFGS Impact:
- **Multiple evaluations** during line search means more opportunities to hit ill-conditioned matrices
- **Parameter updates between closure calls** can push the model into invalid regions
- **Cholesky decomposition** fails more frequently because it's called more often

#### Adam Impact:
- **Single evaluation** per epoch reduces chances of hitting numerical issues
- **Gradual parameter updates** allow the model to stay in valid regions
- **Early stopping** can catch issues before they become severe

### 5. **The Closure Function Problem**

#### Current Closure Implementation:
```python
def closure():
    optimizer.zero_grad()
    output = model(model.train_inputs[0])  # Forward pass
    loss = -mll(output, model.train_targets)  # Computes covariance matrix
    
    if torch.isnan(loss):
        return torch.tensor(1e6, ...)  # Penalty value
    
    loss.backward()
    return loss
```

#### Issues:
1. **Penalty Value Doesn't Help**: Returning `1e6` doesn't prevent LBFGS from trying other step sizes
2. **No Parameter Validation**: The closure doesn't check if parameters are valid before forward pass
3. **Silent Failures**: NaN gradients are zeroed, but parameters might already be invalid
4. **Multiple Failures**: If closure fails multiple times, LBFGS keeps trying, accumulating invalid states

### 6. **Learning Rate and Step Size**

#### LBFGS with `lr=0.1`:
- **Line search** can try step sizes up to `lr` times the gradient magnitude
- **No adaptive scaling** - same learning rate for all parameters
- **Large steps** can push parameters far from valid regions

#### Adam with `lr=0.1`:
- **Adaptive learning rates** per parameter (based on gradient history)
- **Momentum** (betas) provides stability
- **Bounded updates** - step size is always controlled by learning rate

### 7. **Why Your Validation Fails with LBFGS**

The validation checks for NaN/Inf in hyperparameters **after training**:
```python
# Check hyperparameters directly for NaN/Inf
if torch.isnan(noise_val).any() or torch.isinf(noise_val).any():
    has_invalid = True
```

#### What Happens:
1. LBFGS training completes (or early stops)
2. During training, parameters may have become invalid
3. The **best model state** (lowest loss) might have been saved when parameters were valid
4. But **final parameters** after all LBFGS steps might be invalid
5. Validation checks final parameters → finds NaN/Inf → marks as invalid

#### Why Adam Works:
1. Adam training completes
2. Parameters remain valid throughout (gradual updates)
3. Best model state has valid parameters
4. Final parameters are also valid
5. Validation passes

## Solutions and Recommendations

### 1. **Increase Jitter for LBFGS**
```python
jitter = 1e-4  # Instead of 1e-6 for LBFGS
```
This helps with numerical stability during Cholesky decomposition.

### 2. **Add Parameter Constraints**
```python
# In model initialization
model.covar_module.base_kernel.raw_lengthscale_constraint = gpytorch.constraints.Positive()
model.covar_module.raw_outputscale_constraint = gpytorch.constraints.Positive()
model.likelihood.raw_noise_constraint = gpytorch.constraints.GreaterThan(1e-4)
```

### 3. **Reduce LBFGS Learning Rate**
```python
lr = 0.01  # Instead of 0.1
```
Smaller steps reduce chances of invalid parameter regions.

### 4. **Reduce max_iter for LBFGS**
```python
max_iter = 10  # Instead of 20
```
Fewer closure calls = fewer opportunities for numerical issues.

### 5. **Add Parameter Validation in Closure**
```python
def closure():
    # Validate parameters before forward pass
    for param in model.parameters():
        if torch.isnan(param).any() or torch.isinf(param).any():
            return torch.tensor(1e6, ...)
    
    optimizer.zero_grad()
    # ... rest of closure
```

### 6. **Use Adam (Current Solution)**
Adam works because:
- Single forward pass per step
- Adaptive, bounded updates
- More stable for GP training
- Better handles numerical instabilities

## Conclusion

**LBFGS fails** because its line search algorithm performs multiple forward passes per step, repeatedly evaluating the GP covariance matrix. This exposes numerical instabilities that cause Cholesky decomposition to fail, leading to NaN values in parameters.

**Adam works** because it performs only one forward/backward pass per step, with adaptive learning rates that provide more stable parameter updates. This keeps the model in valid parameter regions and avoids the numerical instability cascade.

For GP training with potential numerical issues (high-dimensional problems, small training sets, extreme hyperparameters), **Adam is more robust** than LBFGS, even though LBFGS can converge faster when it works.

