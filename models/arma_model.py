# Modeling gas prices using ARMA models
# We will model all ARMA combinations up to ARMA(4,4)

# Import libraries
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy import stats
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.graphics.api import qqplot
from statsmodels.graphics.tsaplots import plot_acf

# Data is all loaded and cleaned in the data_cleaning.py file

# Plot Autocorrelation Function (ACF) to determine optimal ARMA combination
plot_acf(forecasting_data.gas_prices.dropna())
plt.show()

# Fit the models
def fitARMA(series,p,q):
    '''
    A function to fit an ARMA model for a given p and q
    Input: Series to model, p, q
    Outpu: Fit Model
    '''

    model = ARIMA(series.dropna(), order=(p,0,q))
    return model

for p in range(0,5):
    for q in range (0, 5):
        mod = fitARMA(forecasting_data.gas_prices,p,q).fit()
        print('-------')
        print(f'ARMA({p},{q}) Model')
        print(f'AIC:{mod.aic}, BIC: {mod.bic}')
        print(f'Param Results:')
        coefs = pd.merge(pd.DataFrame(mod.params),pd.DataFrame(mod.pvalues), left_index=True, right_index=True, how='inner').set_axis(['Param', 'P-Val'], axis=1)
        print(coefs)
