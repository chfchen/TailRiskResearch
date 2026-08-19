# S&P 500 Tail-Risk Analysis: From Static Distributions to Dynamic Forecasting

This repository contains a continuing research project on **downside tail risk in the S&P 500**, progressing from static distribution-based risk modeling to dynamic, time-varying risk forecasting.

The research is organized around two studies:

1. **Static Tail-Risk Modeling:** *Beyond Normality: Comparative Tail-Risk Analysis of S&P 500 Returns*
   publication: https://jhss.scholasticahq.com/article/162826-beyond-normality-comparative-tail-risk-analysis-of-s-p-500-returns
3. **Dynamic Tail-Risk Forecasting:** a follow-up study comparing Historical Simulation, GARCH, GJR-GARCH, and Quantile Regression for one-day-ahead Value at Risk (VaR) forecasting.

## Objectives

The overall objective is to investigate how accurately statistical models measure and forecast extreme downside risk in the U.S. equity market.

The first study examines whether the commonly used normal distribution adequately represents S&P 500 returns and compares it with alternative heavy-tailed models.

The follow-up study extends this research by asking a different question:

> **Can models that account for time-varying volatility and changing market conditions provide more accurate dynamic tail-risk forecasts?**

Together, the studies examine the progression from **static risk estimation** to **dynamic risk forecasting**.

## Data Source

The analyses use the **SPDR S&P 500 ETF Trust (SPY)** as a proxy for the S&P 500.

* **Source:** Yahoo Finance
* **Data:** Daily adjusted closing prices
* **Primary sample:** January 1996 – December 2025
* **Return measure:** Daily logarithmic returns

Using the same underlying market and return series allows the static and dynamic studies to be compared within a consistent framework.

## Methods

### Study 1: Static Distribution-Based Tail Risk

The first study examines the statistical distribution of S&P 500 returns and compares several approaches to modeling downside tail risk:

* Normal Distribution
* Student's t-Distribution
* Two-Component Gaussian Mixture Model (GMM-2)
* Extreme Value Theory (EVT)

The analysis includes:

* Descriptive statistics
* Q-Q plots and distribution fitting
* Normality testing
* Akaike Information Criterion (AIC)
* Bayesian Information Criterion (BIC)
* Value at Risk (VaR)
* Expected Shortfall (ES)
* Exceedance backtesting
* Annual calibration analysis
* Rolling-window analysis
* Market-regime and volatility-condition comparisons

### Study 2: Dynamic Tail-Risk Forecasting

The second study shifts the focus from fitting unconditional return distributions to **forecasting one-day-ahead downside risk as market conditions change**.

The primary models include:

* Historical Simulation
* GARCH(1,1) with Student's t innovations
* GJR-GARCH(1,1) with Student's t innovations
* Rolling Linear Quantile Regression

The primary analysis uses a **1,250-trading-day rolling estimation window** and forecasts **one-day-ahead 99% VaR**.

Model performance is evaluated using:

* VaR violation frequency
* Observed-to-expected violation ratio
* Kupiec unconditional coverage test
* Christoffersen independence test
* Conditional coverage test
* Quantile loss
* Annual and market-regime performance
* Crisis-period performance
* Rolling- versus expanding-window sensitivity

## Key Findings

### Static Tail-Risk Analysis

The first study finds that S&P 500 daily returns differ substantially from the normal distribution, exhibiting **negative skewness, a pronounced central peak, and heavy tails**.

The Normal model performs reasonably under moderate market conditions but increasingly underestimates downside risk at deeper tail levels and during periods of market stress.

Heavy-tailed approaches generally improve tail-risk representation:

* The **Student's t-distribution** provides a strong overall balance between simplicity and tail-risk representation.
* **EVT** performs particularly well for rare and extreme losses.
* Model performance changes across volatility conditions, market regimes, and tail depths.
* **No single static model consistently performs best under all conditions.**

An important conclusion from the rolling-window analysis is that changing the assumed return distribution alone is insufficient to fully capture dynamic financial tail risk.

### Dynamic Tail-Risk Forecasting

The follow-up research investigates this limitation directly by allowing risk forecasts to respond to changing volatility.

Preliminary analysis indicates that conditional-volatility models can respond more effectively to volatility clustering than static distribution-based approaches. The comparison between GARCH and GJR-GARCH also examines whether accounting for the asymmetric response of volatility to positive and negative market shocks improves downside-risk forecasting.

The study further investigates whether improvements in statistical VaR calibration remain consistent across calm, volatile, and crisis market environments.

## Limitations and Potential Biases

Several limitations should be considered when interpreting the results.

### Market Scope

The analysis focuses exclusively on **SPY and the U.S. large-cap equity market**. Results may not generalize to individual stocks, international equity markets, bonds, commodities, cryptocurrencies, or other asset classes.

### Historical Dependence

All models are estimated from historical market data. Structural changes in market behavior may therefore reduce the ability of past relationships to represent future conditions.

### Model Risk

Each model imposes different assumptions. Normal and Student's t models assume particular distributional forms, while GARCH-family models impose specific structures on conditional volatility. Results may therefore depend partly on model specification rather than the underlying market process alone.

### Window Selection

Rolling-window results depend on the selected estimation-window length. A shorter window may respond more rapidly to changing conditions but contain less information, while a longer window may provide more stable estimates but respond more slowly to regime changes.

### Extreme Events

Rare events provide relatively few observations for statistical evaluation. Conclusions regarding very deep tail risk are therefore inherently less certain than conclusions about more common market behavior.

### Backtesting Limitations

Passing a statistical backtest does not prove that a model accurately represents the true underlying risk process. Similarly, failure of a particular backtest does not necessarily mean that a model has no practical value.

### Look-Ahead and Researcher Bias

Dynamic forecasts are designed to use only information available before each forecast date to avoid look-ahead bias. However, model selection, window lengths, confidence levels, and evaluation metrics are research design choices and may introduce researcher-selection bias.

## Research Progression

The two studies represent a progression in the central research question:

**Static modeling**

> Which statistical distributions best represent S&P 500 downside tail risk?

↓

**Dynamic forecasting**

> How does tail risk change over time, and can conditional-volatility models forecast those changes more accurately?

This progression reflects a broader objective of moving from describing historical return distributions toward understanding and forecasting financial risk under changing market conditions.

## Disclaimer

This repository is intended for **academic and educational research purposes only**. The analyses, models, and results presented here should not be interpreted as investment advice or recommendations to buy or sell any financial asset.
