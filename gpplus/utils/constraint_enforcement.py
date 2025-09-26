"""
Constraint enforcement utilities for GPyTorch models.
"""

import torch
import logging

logger = logging.getLogger(__name__)

def enforce_parameter_constraints(model):
    """
    Enforce parameter constraints after optimizer steps.
    
    This is necessary because GPyTorch constraints are not automatically
    enforced during optimization - the optimizer can push parameters outside
    the constraint bounds.
    
    Args:
        model: The model whose parameters need constraint enforcement
    """
    try:
        logger.info("Enforcing parameter constraints...")
        constrained_count = 0
        for name, param in model.named_parameters():
            if not param.requires_grad:
                continue
                
            if hasattr(param, 'constraint') and param.constraint is not None:
                # Check if parameter has bounds
                if hasattr(param.constraint, 'lower_bound') and hasattr(param.constraint, 'upper_bound'):
                    # Clamp parameter to constraint bounds
                    old_data = param.data.clone()
                    param.data = torch.clamp(param.data, param.constraint.lower_bound, param.constraint.upper_bound)
                    
                    # Log if parameter was clamped
                    if not torch.allclose(old_data, param.data):
                        constrained_count += 1
                        logger.info(f"CONSTRAINED parameter {name}: {old_data.item() if old_data.numel() == 1 else old_data.flatten().tolist()} -> {param.data.item() if param.data.numel() == 1 else param.data.flatten().tolist()}")
        
        # Also enforce constraints on GPyTorch module parameters
        for name, module in model.named_modules():
            if 'gpytorch' in module.__class__.__module__:
                # Enforce noise constraint
                if hasattr(module, 'noise_covar') and hasattr(module.noise_covar, 'raw_noise'):
                    raw_noise = module.noise_covar.raw_noise
                    # Check both parameter-level and module-level constraints
                    constraint = None
                    if hasattr(raw_noise, 'constraint') and raw_noise.constraint is not None:
                        constraint = raw_noise.constraint
                    elif hasattr(module.noise_covar, 'raw_noise_constraint') and module.noise_covar.raw_noise_constraint is not None:
                        constraint = module.noise_covar.raw_noise_constraint
                    
                    if constraint is not None and hasattr(constraint, 'lower_bound') and hasattr(constraint, 'upper_bound'):
                        old_data = raw_noise.data.clone()
                        raw_noise.data = torch.clamp(raw_noise.data, constraint.lower_bound, constraint.upper_bound)
                        if not torch.allclose(old_data, raw_noise.data):
                            constrained_count += 1
                            logger.info(f"CONSTRAINT ENFORCED: Noise {old_data.item():.6f} -> {raw_noise.data.item():.6f} (bounds: [{constraint.lower_bound}, {constraint.upper_bound}])")
                        else:
                            logger.debug(f"Noise already in bounds: {raw_noise.data.item():.6f}")
                
                # Enforce outputscale constraint
                if hasattr(module, 'raw_outputscale'):
                    raw_outputscale = module.raw_outputscale
                    # Check both parameter-level and module-level constraints
                    constraint = None
                    if hasattr(raw_outputscale, 'constraint') and raw_outputscale.constraint is not None:
                        constraint = raw_outputscale.constraint
                    elif hasattr(module, 'raw_outputscale_constraint') and module.raw_outputscale_constraint is not None:
                        constraint = module.raw_outputscale_constraint
                    
                    if constraint is not None and hasattr(constraint, 'lower_bound') and hasattr(constraint, 'upper_bound'):
                        old_data = raw_outputscale.data.clone()
                        raw_outputscale.data = torch.clamp(raw_outputscale.data, constraint.lower_bound, constraint.upper_bound)
                        if not torch.allclose(old_data, raw_outputscale.data):
                            constrained_count += 1
                            logger.info(f"CONSTRAINED outputscale: {old_data.item():.6f} -> {raw_outputscale.data.item():.6f}")
                
                # Enforce lengthscale constraint
                if hasattr(module, 'raw_lengthscale'):
                    raw_lengthscale = module.raw_lengthscale
                    # Check both parameter-level and module-level constraints
                    constraint = None
                    if hasattr(raw_lengthscale, 'constraint') and raw_lengthscale.constraint is not None:
                        constraint = raw_lengthscale.constraint
                    elif hasattr(module, 'raw_lengthscale_constraint') and module.raw_lengthscale_constraint is not None:
                        constraint = module.raw_lengthscale_constraint
                    
                    if constraint is not None and hasattr(constraint, 'lower_bound') and hasattr(constraint, 'upper_bound'):
                        old_data = raw_lengthscale.data.clone()
                        raw_lengthscale.data = torch.clamp(raw_lengthscale.data, constraint.lower_bound, constraint.upper_bound)
                        if not torch.allclose(old_data, raw_lengthscale.data):
                            constrained_count += 1
                            logger.info(f"CONSTRAINED lengthscale: {old_data.flatten().tolist()} -> {raw_lengthscale.data.flatten().tolist()}")
        
        logger.info(f"Constraint enforcement completed. {constrained_count} parameters were constrained.")
                                
    except Exception as e:
        logger.warning(f"Error enforcing parameter constraints: {e}")

def create_constraint_enforcement_callback():
    """
    Create a callback function that can be used in training loops
    to enforce parameter constraints after each optimizer step.
    
    Returns:
        callable: A function that takes (model, optimizer, epoch) as arguments
    """
    def constraint_callback(model, optimizer, epoch):
        enforce_parameter_constraints(model)
    
    return constraint_callback
