Stock Price Analysis and Monte Carlo Simulation
Overview

This project is dedicated to analyzing historical stock price data and forecasting future prices using Monte Carlo simulations. The main focus is on the stock of Home Depot (HD), but the methods used can be applied to any stock.
This analysis includes visualizing investment growth over time, calculating log returns and their volatility, and predicting future stock prices and option valuations.

Features

    Data Loading and Cleaning: Load historical stock price data, rename columns for clarity, and calculate log returns.
    Investment Growth Visualization: Plot the growth of $1 invested in the stock over time, showing the cumulative return.
    Statistical Analysis: Calculate the average daily and annual log return and the volatility of the stock.
    Monte Carlo Simulation: Forecast future stock prices using the Geometric Brownian Motion model, a common approach in finance for modeling stock prices.
    Options Pricing: Calculate the payoffs and prices for call and put options based on the simulated future stock prices.

Dependencies

    Python 3.x
    Libraries: pandas, numpy, datetime, scipy, matplotlib

File Descriptions

    data-home-depo.csv: The dataset containing historical stock prices for Home Depot.
    stock_analysis.py: The main Python script with all the analysis and simulations.

How to Run

    Ensure all dependencies are installed.
    Place the data-home-depo.csv file in the same directory as the script.
    Run the script using Python: python stock_analysis.py.

Detailed Description
Data Preparation

    The historical stock data is loaded from a CSV file.
    Relevant columns are renamed for clarity and ease of use.
    Logarithmic returns are calculated to understand the relative changes in stock price.

Growth Visualization Function

    A function is created to plot the cumulative returns, showing how an investment in the stock would grow over time.
    The x-axis is formatted to show only the year for a cleaner look.

Statistical Analysis

    The script calculates the mean and standard deviation of the daily log returns.
    These statistics are annualized to provide a long-term perspective.

Monte Carlo Simulation

    A Monte Carlo simulation is conducted to forecast future stock prices.
    The Geometric Brownian Motion model is used, incorporating drift and volatility estimated from historical data.
    The simulation generates multiple potential future paths for the stock price, providing a distribution of possible outcomes.

Options Pricing

    Using the results of the Monte Carlo simulation, the script calculates the payoffs for European call and put options based on a given strike price.
    The average payoffs are discounted back to present value, giving an estimate of the options' prices.

Results and Interpretation

    The script prints out key statistics, including the mean and volatility of log returns, and the average prices of call and put options.
    Plots are generated to visualize the cumulative return of the investment and the distribution of possible future stock prices.