
import pandas as pd
import numpy as np
from scipy.optimize import minimize
import matplotlib.pyplot as plt

# Create the dataset
data = pd.DataFrame({
    'country': ['Nigeria']*6 + ['Ghana']*6 + ['Kenya']*6,
    'year': list(range(2016, 2022))*3,
    'AIDI': [23.5, 24.1, 24.8, 25.3, 25.7, 26.2,
             28.7, 29.2, 29.8, 30.3, 30.9, 31.4,
             25.4, 26.0, 26.5, 27.1, 27.6, 28.2],
    'FDI_inflows': [4449, 3503, 1997, 3308, 2385, 3310,
                    3477, 3255, 3007, 3879, 1880, 2530,
                    393, 671, 1626, 1334, -5, 1099]
})

# Define the negative log-likelihood function
def neg_log_likelihood(params):
    alpha, beta, sigma = params
    mu = alpha + beta * data['AIDI']
    residuals = data['FDI_inflows'] - mu
    log_lik = -np.sum(-0.5 * np.log(2 * np.pi * sigma**2) - (residuals**2) / (2 * sigma**2))
    return -log_lik

# Initial guesses
init_params = [0, 0, 1]

# Optimize using BFGS
result = minimize(neg_log_likelihood, init_params, method='BFGS')

# Output results
print("MLE Estimates:")
print(f"Alpha (Intercept): {result.x[0]:.4f}")
print(f"Beta (Slope on AIDI): {result.x[1]:.4f}")
print(f"Sigma (Std Dev): {abs(result.x[2]):.4f}")

# Predict FDI inflows
data['FDI_pred'] = result.x[0] + result.x[1] * data['AIDI']

# Plot actual vs predicted FDI inflows
plt.figure(figsize=(10, 6))
for country in data['country'].unique():
    subset = data[data['country'] == country]
    plt.plot(subset['year'], subset['FDI_inflows'], marker='o', label=f"{country} Actual")
    plt.plot(subset['year'], subset['FDI_pred'], linestyle='--', label=f"{country} Predicted")

plt.title('Actual vs Predicted FDI Inflows (MLE Estimates)')
plt.xlabel('Year')
plt.ylabel('FDI Inflows (USD Millions)')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
