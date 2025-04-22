import numpy as np
import matplotlib.pyplot as plt

# -----------------------------
# Step 1: Generate synthetic data
# -----------------------------
np.random.seed(42)  # for reproducibility

true_intercept = 1.0
true_slope = 2.0
sigma_e = 3.0  # known noise

sample_size = 500
x = np.random.uniform(0, 10, sample_size)
noise = np.random.normal(0, sigma_e, sample_size)
y = true_slope * x + true_intercept + noise

# -----------------------------
# Step 2: Initialize prior beliefs
# -----------------------------
a_0 = 0.5  # initial belief for intercept
b_0 = 0.5  # initial belief for slope
sigma_a_0 = 0.5
sigma_b_0 = 0.5

# Prior mean vector
beta_0 = np.array([[a_0], [b_0]])

# Prior covariance matrix
sigma_beta_0 = np.array([
    [sigma_a_0**2, 0],
    [0, sigma_b_0**2]
])

# -----------------------------
# Step 3: Bayesian recursive update
# -----------------------------
beta_recorder = [beta_0]  # record the history of beta updates

for pair in range(sample_size // 2):
    # Extract two points at a time
    x1, x2 = x[2 * pair], x[2 * pair + 1]
    y1, y2 = y[2 * pair], y[2 * pair + 1]

    # Calculate mean and covariance of observed data
    mu_y = np.array([[(x1 * y2 - x2 * y1) / (x1 - x2)],
                     [(y1 - y2) / (x1 - x2)]])

    sigma_y = np.array([
        [(x1 / (x1 - x2))**2 + (x2 / (x1 - x2))**2, 0],
        [0, 2 * (1 / (x1 - x2))**2]
    ]) * sigma_e**2

    # Bayesian update: combine prior with observed data
    inv_sigma_prior = np.linalg.inv(sigma_beta_0)
    inv_sigma_obs = np.linalg.inv(sigma_y)

    sigma_beta_1 = np.linalg.inv(inv_sigma_prior + inv_sigma_obs)
    beta_1 = sigma_beta_1 @ (inv_sigma_prior @ beta_0 + inv_sigma_obs @ mu_y)

    # Update for next iteration
    beta_0 = beta_1
    sigma_beta_0 = sigma_beta_1
    beta_recorder.append(beta_0)

# -----------------------------
# Step 4: Print final parameters
# -----------------------------
final_intercept = beta_0[0, 0]
final_slope = beta_0[1, 0]
print(f"Estimated parameters: Intercept = {final_intercept:.7f}, Slope = {final_slope:.7f}")

# -----------------------------
# Step 5: Plot regression line evolution
# -----------------------------
xfit = np.linspace(0, 10, sample_size)
ytrue = true_slope * xfit + true_intercept

plt.figure(figsize=(10, 6))
plt.plot(xfit, ytrue, label="True Line", linewidth=3)

# Show updates at different stages
stages = [0, 1, 10, 100]
for i in stages:
    intercept_i = beta_recorder[i][0, 0]
    slope_i = beta_recorder[i][1, 0]
    y_stage = slope_i * xfit + intercept_i
    label = f"{i}th update" if i != 0 else "Initial Belief"
    plt.plot(xfit, y_stage, label=label, linewidth=1)

plt.title("Bayesian Linear Regression Dynamics")
plt.xlabel("x")
plt.ylabel("y")
plt.legend()
plt.grid(True)
plt.show()
