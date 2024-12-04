import torch
import gpytorch
import gpplus
import numpy as np
import matplotlib.pyplot as plt
import logging




# Example usage
# Set random seed for reproducibility
torch.manual_seed(42)

# Generate training data using Sobol sequence and reshape to be 2D
sobol_engine = torch.quasirandom.SobolEngine(dimension=1)
train_x = sobol_engine.draw(12).squeeze(1)
train_y = torch.sin(train_x * (2 * torch.pi)) + torch.randn(train_x.size()) * 0.05

# Generate test data and reshape to be 2D
test_x = torch.linspace(0, 1, 51)
test_y = torch.sin(test_x * (2 * torch.pi))  # True test data without noise for comparison

# Define components for the GP model
mean = gpytorch.means.ConstantMean()
kernel = gpytorch.kernels.RBFKernel()
likelihood = gpytorch.likelihoods.GaussianLikelihood()

# Create model
model = gpplus.models.GPModel(train_x, train_y, likelihood, mean, kernel)

# Instantiate trainer
trainer = gpplus.optim.GPTrainer_v4(model=model, train_x=train_x, train_y=train_y, num_epochs=500, initialize_params=True, seed=22)

results = trainer.multiple_process([42, 43, 44, 45])


for result in results:

    print(result)


# Function to plot GP predictions
def plot_gp_predictions(train_x: torch.Tensor, train_y: torch.Tensor,
                        test_x: torch.Tensor, test_y: torch.Tensor,
                        mean: torch.Tensor, lower: torch.Tensor, upper: torch.Tensor):
    # Convert tensors to numpy for plotting
    test_x_np = test_x.numpy()
    mean_np = mean.numpy()
    lower_np = lower.numpy()
    upper_np = upper.numpy()

    # Plotting
    plt.figure(figsize=(12, 6))
    plt.plot(test_x_np, mean_np, 'b', label='Predicted Mean')
    plt.fill_between(test_x_np, lower_np, upper_np, alpha=0.3, label='Confidence Interval')
    plt.scatter(train_x.numpy(), train_y.numpy(), c='r', label='Training Data')
    plt.scatter(test_x_np, test_y.numpy(), c='g', label='Test Data', marker='x')
    plt.legend()
    plt.title("GP Model Predictions")
    plt.xlabel("Input")
    plt.ylabel("Output")
    plt.show()

'''
# Iterate over each state_dict in the results, load it to the model, and plot predictions
for state_dict in results:
    model.load_state_dict(state_dict)
    model.eval()

    # Make predictions
    with torch.no_grad(), gpytorch.settings.fast_pred_var():
        observed_pred = model(test_x)
        mean = observed_pred.mean
        lower, upper = observed_pred.confidence_region()

    # Plot the predictions
    plot_gp_predictions(train_x, train_y, test_x, test_y, mean, lower, upper)
'''

'''
# Iterate over each state_dict in the results, load it to the model, and plot predictions
for result in results:
    print(result)
    print("#####################")
    
    state_dict = value if isinstance(value, dict) else value['state_dict']
    model.load_state_dict(state_dict)
    model.eval()

    # Make predictions
    with torch.no_grad(), gpytorch.settings.fast_pred_var():
        observed_pred = model(test_x)
        mean = observed_pred.mean
        lower, upper = observed_pred.confidence_region()

    # Plot the predictions
    plot_gp_predictions(train_x, train_y, test_x, test_y, mean, lower, upper)
    '''

# Iterate over each result, load the state_dict to the model, and plot predictions
for result in results:
    #print(result['state_dict'])
    
    state_dict = result['state_dict']
    
    model.load_state_dict(state_dict)
    # Set the model to evaluation mode
    model.eval()
    #self.likelihood.eval()

    with torch.no_grad(), gpytorch.settings.fast_pred_var():
        # Make predictions
        observed_pred = model.likelihood(model(test_x))

        # Get the mean, lower and upper confidence bounds
        mean = observed_pred.mean
        lower, upper = observed_pred.confidence_region()

        #logger.info("Evaluation completed.")
        #return mean, lower, upper

    # Plot the predictions
    plot_gp_predictions(train_x, train_y, test_x, test_y, mean, lower, upper)
    
