import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
import seaborn as sns

# Simulate panel data for FDI and AfDB indices
np.random.seed(42)
countries = ['Kenya', 'Nigeria', 'Ethiopia', 'Rwanda', 'Ghana', 'Uganda', 'Zambia', 'Tanzania', 'Senegal', "Cote d'Ivoire"]
years = list(range(2016, 2022))

data = []
for country in countries:
    for year in years:
        aidi = np.random.uniform(10, 70)
        tci = np.random.uniform(5, 30)
        eci = np.random.uniform(5, 30)
        ici = np.random.uniform(5, 30)
        wssi = np.random.uniform(5, 30)
        error = np.random.normal(0, 5)
        fdi = 0.36 * aidi + 0.27 * tci + 0.09 * eci + 0.34 * ici + 0.20 * wssi + 13 + error
        data.append([country, year, fdi, aidi, tci, eci, ici, wssi])

df = pd.DataFrame(data, columns=['Country', 'Year', 'FDI', 'AIDI', 'TCI', 'ECI', 'ICI', 'WSSI'])

# Linear regression using Maximum Likelihood Estimation
from statsmodels.formula.api import ols
model = ols("FDI ~ AIDI + TCI + ECI + ICI + WSSI", data=df).fit()
print(model.summary())

# Extrapolate FDI for 2022–2030
grouped = df.groupby('Year')[['FDI']].mean().reset_index()
X_train = grouped['Year'].values.reshape(-1, 1)
y_train = grouped['FDI'].values

lin_reg = LinearRegression().fit(X_train, y_train)
future_years = np.arange(2022, 2031).reshape(-1, 1)
y_future_pred = lin_reg.predict(future_years)

# Counterfactual: +20% more FDI
y_counterfactual = y_future_pred * 1.20

# Plot the results
plt.figure(figsize=(10, 6))
plt.scatter(X_train, y_train, label='Historical FDI (2016–2021)', color='blue')
plt.plot(np.arange(2016, 2031), lin_reg.predict(np.arange(2016, 2031).reshape(-1, 1)), linestyle='--', label='MLE Trend', color='black')
plt.plot(future_years, y_future_pred, label='Forecast FDI (2022–2030)', color='green')
plt.plot(future_years, y_counterfactual, label='Counterfactual FDI (+20%)', color='red')
plt.xlabel('Year')
plt.ylabel('Average FDI')
plt.title('FDI Projection with Counterfactual Scenario')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
