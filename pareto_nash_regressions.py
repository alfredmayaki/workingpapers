# pareto_nash_regressions.py

import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf
from linearmodels.panel import PanelOLS

# ------------------------
# 1. GLM Regression (Cross-Sectional)
# ------------------------

# Load GLM dataset
glm_df = pd.read_csv("glm_data.csv")

# Fit GLM model using Gaussian identity link
glm_model = smf.glm(
    formula="growth ~ endowment + age + export_intensity + phi",
    data=glm_df,
    family=sm.families.Gaussian()
).fit()

print("\n=== GLM Regression Results ===")
print(glm_model.summary())

# ------------------------
# 2. Fixed Effects Panel Regression
# ------------------------

# Load panel dataset
panel_df = pd.read_csv("panel_data.csv")

# Set panel index
panel_df = panel_df.set_index(["firm_id", "year"])

# Add constant term
panel_df["const"] = 1

# Fit Fixed Effects model
fe_model = PanelOLS.from_formula(
    "growth ~ const + endowment + export_intensity + phi + age + EntityEffects",
    data=panel_df
).fit()

print("\n=== Fixed Effects Panel Regression Results ===")
print(fe_model.summary)
