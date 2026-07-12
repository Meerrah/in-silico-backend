import pandas as pd
import numpy as np

# Set a random seed for reproducibility
np.random.seed(42)

# Generate 5000 synthetic patient records
n_samples = 5000
data = {
    "ast": np.random.normal(35, 10, n_samples),  # Liver enzymes
    "crp": np.random.normal(2.5, 1.2, n_samples), # Inflammation markers
    "dosage": np.random.uniform(20, 100, n_samples) # Drug dosage in mg
}

# Convert to a Pandas DataFrame and save as CSV
df = pd.DataFrame(data)

# Ensure no negative biological values exist
df['ast'] = df['ast'].clip(lower=5)
df['crp'] = df['crp'].clip(lower=0.1)

df.to_csv("phase_1_trials.csv", index=False)
print("✅ Generated phase_1_trials.csv with 5,000 patient records!")
