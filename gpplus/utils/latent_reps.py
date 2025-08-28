import torch
import torch.nn.functional as F

def get_latent_representations(model, data, qual_dict=None, input_type='categorical'):
    """
    Generate latent representations with robust column index handling
    
    Args:
        model: Trained GP model
        data: Data dictionary containing metadata
        qual_dict: Original qualitative variable specification
        input_type: 'categorical' or 'source'
        
    Returns:
        tuple: (all_combinations, latent_reps)
            - all_combinations: One-hot encoded combinations or source indices
            - latent_reps: Latent space representations
            If no categorical/source variables, returns (None, None)
    """
    model.eval()
    combined_kernel = model.covar_module
    
    if input_type == 'categorical':
        # Check if categorical variables exist
        if not hasattr(combined_kernel, 'cat_encoder') or not qual_dict:
            return None, None, None
            
        latent_net = combined_kernel.cat_encoder
        # Get the actual categorical dimensions from qual_dict
        # We need to maintain the original ordering of categorical variables
        cat_vars = sorted(qual_dict.items())  # Sort by original column index
        cat_dims = [dim for _, dim in cat_vars]  # Extract just the dimensions
        
        # Generate all possible combinations
        indices = torch.cartesian_prod(*[torch.arange(dim) for dim in cat_dims])
        
        # Create one-hot encoded combinations
        one_hots = [F.one_hot(indices[:, i], num_classes=dim) for i, dim in enumerate(cat_dims)]
        all_combinations = torch.cat(one_hots, dim=1)
    else:  # source
        # Check if source variables exist
        if not hasattr(combined_kernel, 'source_encoder') or len(data['metadata']['source_names']) <= 1:
            return None, None
            
        latent_net = combined_kernel.source_encoder
        num_sources = len(data['metadata']['source_names'])
        all_combinations = torch.eye(num_sources)
    
    # Get latent representations
    with torch.no_grad():
        latent_reps = latent_net(all_combinations.float())
    if input_type == 'categorical':
        return all_combinations, indices, latent_reps
    else:
        return all_combinations, latent_reps