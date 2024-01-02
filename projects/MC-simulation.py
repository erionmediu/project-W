# Importing needed packages 

import pandas as pd
import numpy as np
import datetime 
import scipy.stats as sts
import matplotlib.pyplot as plt
import math
import scipy.optimize
from scipy.stats import norm
import matplotlib.pyplot as plt
import matplotlib.dates as mdates


# Loading the data 

data = pd.read_csv("data-home-depo.csv")
data.rename(columns={'RETX':'ret', 'PRC':'prc'}, inplace=True)
data['log-ret'] = np.log(1+data['ret'])
data.drop(columns='PERMNO', inplace = True)
print(data)


def plot_financial_data(data):
    data['date'] = pd.to_datetime(data['date'], format='%d/%m/%Y')
    # Calculate the cumulative return for $1 invested
    data['cumulative_return'] = (1 + data['ret']).cumprod()
    # Plotting
    fig, ax = plt.subplots(figsize=(12, 6))
    # Plotting the cumulative returns
    ax.plot(data['date'], data['cumulative_return'], color='blue')
    ax.set_title('Growth of $1 Investment Over Time')
    ax.set_xlabel('Date')
    ax.set_ylabel('Cumulative Return')
    # Format the x-axis to show only the year
    ax.xaxis.set_major_locator(mdates.YearLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
    # Rotate the date labels for better readability
    plt.xticks(rotation=45)
    plt.show()

plot_financial_data(data)

########

# average daily log return and volatility of the stock 
log_col = data['log-ret']
prc_col = data['prc']
ret_col = data['ret']

mean = log_col.mean()
std = np.std(log_col)
print(round(mean , 4 ) ,round(std, 4))

#average anual log return and volatility of the stock assuming 252 trading days
anlogmean = ((mean+1) ** 252)-1
print(round(anlogmean,4 ))
anstdlog = std*math.sqrt(252)
print(round(anstdlog,4))


### Set up simulation parameters 
Szero = prc_col.iloc[-1] # last price date: 30/12/2022
print(Szero)

annualrf = 0.03 # assumed at 3 percent
dailyrf = annualrf/252 

mc_sims = 10000 # number of simulations
T = 19 #timeframe of 19 days chosen

# Parameters
# drift coefficent
drift_rf = dailyrf
# number of steps
step = 19
# time in years just in case extra information
years = step/252
# number of sims
simulation_number = 10000
# initial stock price
S0 = Szero
# volatility
volatility = std

# calc each time step
dt = 1

# simulation using numpy arrays
random_gbm = np.random.normal(0, np.sqrt(dt), size=(simulation_number,step)).T
St = np.exp(
    (drift_rf - volatility ** 2 / 2) * dt
    + volatility * random_gbm
)
# include array of 1's
St = np.vstack([np.ones(simulation_number), St])
St.shape
# multiply through by S0 and return the cumulative product of elements along a given simulation path (axis=0). 
St = S0 * St.cumprod(axis=0)


# Define time interval correctly 
time = np.linspace(0,T,step+1)
# Require numpy array that is the same shape as St
tt = np.full(shape=(simulation_number,step+1), fill_value=time).T
plt.plot(tt, St)
plt.xlabel("Days $(t)$")
plt.ylabel("Stock Price $(S_t)$")
plt.title(
    "MC simulations of Geometric Brownian Motion\n $dS_t = \mu S_t dt + \sigma S_t dW_t$\n $S_0 = {0}, \mu = {1}, \sigma = {2}$".format(S0, round(drift_rf,4), round(volatility,4))
)
plt.xlim([0,19])
plt.xticks([0, 1, 2, 3, 4, 5, 6, 7, 8 , 9, 10 , 11, 12, 13, 14, 15, 16, 17, 18, 19])
plt.show()
print('The Mean is ' + str(np.mean(St)))


# Create empty list to store the last prices of each simulation
last_St = St[-1]
last_St_list = np.array(last_St).tolist()



dif_prices = [prices - 315.85999 for prices in last_St_list] # 315.859 is the price observed at 31/12/2022
Payoff_Calls = [] # Empty list for payoff calls
Payoff_Puts = [] # Empty list for payoff puts

# payoff list loop
for elem in dif_prices: 
    if elem <= 0:
        Payoff_Calls.append(0)
        Payoff_Puts.append(abs(elem))
    else:
        Payoff_Calls.append(elem)
        Payoff_Puts.append(0)

# Zip the two lists together to make one dataframe where payoff values are together
zipped = list(zip(Payoff_Calls, Payoff_Puts))
df_Payoff = pd.DataFrame(zipped, columns=['Payoff Calls', 'Payoff Puts'])
df_Payoff


# Getting number of nonzeros in each column
df_Payoff.astype(bool).sum(axis=0)


# Take the payoff dataframe and create one with new column names
df_price_options = df_Payoff
df_price_options.columns = ['Price Call', 'Price Put']

# Apply formula to both columns
Call_p = df_price_options['Price Call'].mean() * np.exp((-dailyrf) * 19)
Put_p = df_price_options['Price Put'].mean() * np.exp((-dailyrf) * 19)
print (Call_p, Put_p)





################ pipeline 



import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from scipy.stats import norm
import math

def load_and_prepare_data(filepath):
    data = pd.read_csv(filepath)
    data.rename(columns={'RETX':'ret', 'PRC':'prc'}, inplace=True)
    data['log-ret'] = np.log(1 + data['ret'])
    data.drop(columns='PERMNO', inplace=True)
    return data

def plot_financial_data(data):
    data['date'] = pd.to_datetime(data['date'], format='%d/%m/%Y')
    data['cumulative_return'] = (1 + data['ret']).cumprod()
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(data['date'], data['cumulative_return'], color='blue')
    ax.set_title('Growth of $1 Investment Over Time')
    ax.set_xlabel('Date')
    ax.set_ylabel('Cumulative Return')
    ax.xaxis.set_major_locator(mdates.YearLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
    plt.xticks(rotation=45)
    plt.show()

def calculate_statistics(data):
    mean = data['log-ret'].mean()
    std = np.std(data['log-ret'])
    anlogmean = ((mean + 1) ** 252) - 1
    anstdlog = std * math.sqrt(252)
    return mean, std, anlogmean, anstdlog

def monte_carlo_simulation(data, S0, annualrf, T, mc_sims, step):
    dailyrf = annualrf / 252
    drift_rf = dailyrf
    volatility = np.std(data['log-ret'])
    random_gbm = np.random.normal(0, np.sqrt(1), size=(mc_sims, step)).T
    St = np.exp((drift_rf - volatility ** 2 / 2) + volatility * random_gbm)
    St = np.vstack([np.ones(mc_sims), St])
    St = S0 * St.cumprod(axis=0)
    return St

def plot_simulation(St, S0, drift_rf, volatility, T, step):
    time = np.linspace(0, T, step + 1)
    tt = np.full(shape=(mc_sims, step + 1), fill_value=time).T
    plt.plot(tt, St)
    plt.xlabel("Days $(t)$")
    plt.ylabel("Stock Price $(S_t)$")
    plt.title("MC simulations of Geometric Brownian Motion\n $dS_t = \mu S_t dt + \sigma S_t dW_t$\n $S_0 = {}, \mu = {}, \sigma = {}$".format(S0, round(drift_rf,4), round(volatility,4)))
    plt.xlim([0, T])
    plt.xticks(range(T+1))
    plt.show()
    print('The Mean is ' + str(np.mean(St[-1])))

def option_pricing(St, strike_price, dailyrf, T):
    last_St = St[-1]
    dif_prices = [price - strike_price for price in last_St]
    Payoff_Calls = [max(elem, 0) for elem in dif_prices]
    Payoff_Puts = [max(strike_price - price, 0) for price in last_St]
    Call_p = np.mean(Payoff_Calls) * np.exp((-dailyrf) * T)
    Put_p = np.mean(Payoff_Puts) * np.exp((-dailyrf) * T)
    return Call_p, Put_p

def run_pipeline(filepath, S0, annualrf, T, mc_sims, step, strike_price):
    data = load_and_prepare_data(filepath)
    plot_financial_data(data)
    mean, std, anlogmean, anstdlog = calculate_statistics(data)
    print(f"Mean: {round(mean, 4)}, Std: {round(std, 4)}, Annualized Mean: {round(anlogmean, 4)}, Annualized Std: {round(anstdlog, 4)}")
    St = monte_carlo_simulation(data, S0, annualrf, T, mc_sims, step)
    plot_simulation(St, S0, annualrf / 252, std, T, step)
    Call_p, Put_p = option_pricing(St, strike_price, annualrf / 252, T)
    print(f"Call Option Price: {Call_p}, Put Option Price: {Put_p}")

# Parameters
filepath = "data-home-depo.csv"
S0 = 315.85999  # last observed price
annualrf = 0.03  # assumed 3 percent annual risk-free rate
T = 19           # 19 days timeframe
mc_sims = 10000  # number of simulations
step = 19        # number of steps
strike_price = 315.85999  # strike price for options

# Running the pipeline
run_pipeline(filepath, S0, annualrf, T, mc_sims, step, strike_price)


