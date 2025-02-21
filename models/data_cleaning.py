import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
from statsmodels.tsa import stattools

# read in data
path = '/Users/bram.stockbauer/forecasting-gas-prices/data'

crude_oil_prices = pd.read_csv(f'{path}/crude_oil_prices.csv', parse_dates=['observation_date'], names=['observation_date','value'],header=0).set_index('observation_date')
corporate_profits = pd.read_csv(f'{path}/CP.csv', parse_dates=['observation_date'], names=['observation_date','value'],header=0).set_index('observation_date')
gas_prices = pd.read_csv(f'{path}/gas_prices.csv', parse_dates=['observation_date'], names=['observation_date','value'],header=0).set_index('observation_date')
gdp = pd.read_csv(f'{path}/GDP.csv', parse_dates=['observation_date'], names=['observation_date','value'],header=0).set_index('observation_date')
unemp_rate = pd.read_csv(f'{path}/unemployment_rate.csv', parse_dates=['observation_date'], names=['observation_date','value'],header=0).set_index('observation_date')

# Convert monthly data to quarterly
def monthlyToQuarterly(df,aggType = 'mean'):
    '''
    A function to convert monthly data to quarterly
    Inputs: Monthly dataframe
    Outputs: Quarterly dataframe
    '''

    if aggType == 'mean':
        df = df.resample('QS').mean()
        #df.index = df.index - pd.DateOffset(months=2) # Set the quarter to be denoted by the first day rather than the last
        return df
    elif aggType == 'sum':
        df = df.resample('QS').sum()
        #df.index = df.index - pd.DateOffset(months=2) # Set the quarter to be denoted by the first day rather than the last
        return df
    else:
        print('please try again and choose an aggregation (sum or mean)')
        return df


crude_oil_prices = monthlyToQuarterly(crude_oil_prices,aggType='mean')
gas_prices = monthlyToQuarterly(gas_prices, aggType='mean')
unemp_rate = monthlyToQuarterly(unemp_rate, aggType='mean')


# Choose starting date 
def filterDates(df, startDate, endDate):
    '''
    A function to filter down dates to those within a certain time range
    Inputs: A dataframe (with date as index), a start date, and an end date
    Outputs: A dataframe filtered based on the start and end date
    '''

    df = df[(df.index >= startDate) & (df.index <= endDate)]
    return df

crude_oil_prices = filterDates(crude_oil_prices, '1995-01-01', '2022-10-01')
corporate_profits = filterDates(corporate_profits, '1995-01-01', '2022-10-01')
gas_prices = filterDates(gas_prices, '1995-01-01', '2022-10-01')
gdp = filterDates(gdp, '1995-01-01', '2022-10-01')
unemp_rate = filterDates(unemp_rate, '1995-01-01', '2022-10-01')

# Check Data For Stationarity
def adf(series, title):
    '''
    A function to run the ADF test on a time series and check for stationarity
    Input: Series, series title
    Output: ADF test report
    '''

    print(f'ADF Test Report -- {title}')
    test_result = stattools.adfuller(series.dropna(), autolag='AIC')
    labs = ['ADF Test Stat','P-Value','# Lags','# Obs']
    out = pd.Series(test_result[0:4], index=labs)
    for key,val in test_result[4].items():
        out[f'Critical Value {key.title()}']=val
    print(out.to_string())
    if test_result[1] <= 0.05:
        print("Strong evidence against the null hypothesis")
        print("Reject the null hypothesis")
        print("Data has no unit root and is stationary")
    else:
        print("Weak evidence against the null hypothesis")
        print("Fail to reject the null hypothesis")
        print("Data has a unit root and is non-stationary")

adf(crude_oil_prices.value, 'Crude Oil Prices')
adf(corporate_profits.value, 'CP')
adf(gas_prices.value, 'Gas Prices')
adf(gdp.value, 'GDP')
adf(unemp_rate, 'GDP')

# All series except for unemp_rate are stationary. We can solve for this by taking first differences
adf(crude_oil_prices.value.diff(1), 'Crude Oil Prices (first difference)')
crude_oil_prices.value = crude_oil_prices.value.diff(1).dropna()

adf(corporate_profits.value.diff(1), 'CP')
corporate_profits.value = corporate_profits.value.diff(1).dropna()

adf(gas_prices.value.diff(1), 'Gas Prices')
gas_prices.value = gas_prices.value.diff(1).dropna()

adf(gdp.value.diff(1), 'GDP')
gdp.value = gdp.value.diff(1).dropna()

# Merge all dataframes together for organization
forecasting_data = pd.concat([crude_oil_prices, corporate_profits, gas_prices, gdp, unemp_rate], axis=1).set_axis(['crude_oil_prices', 'corporate_profits', 'gas_prices', 'gdp', 'unemp_rate'], axis=1)

