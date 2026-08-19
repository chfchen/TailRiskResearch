# S&P 500 Tail-Risk Analysis: From Static Distributions to Dynamic Forecasting

This repository contains a continuing research project on **downside tail risk in the S&P 500**, progressing from static distribution-based risk modeling to dynamic, time-varying risk forecasting. The overall objective is to investigate how accurately statistical models measure and forecast extreme downside risk in the U.S. equity market.

The research is organized around two studies:

1. **Static Tail-Risk Modeling:**
   
   Scientific question: Which statistical distributions best represent S&P 500 downside tail risk?

   Publication: Beyond Normality: Comparative Tail-Risk Analysis of S&P 500 Returns (https://jhss.scholasticahq.com/article/162826-beyond-normality-comparative-tail-risk-analysis-of-s-p-500-returns)
2. **Dynamic Tail-Risk Forecasting:**

   Scientific question: Which statistical and machine-learning approaches provide the most accurate and reliable dynamic Value-at-Risk forecasts for the S&P 500 across changing market conditions?

   Publication: Dynamic Value-at-Risk Forecasting for the S&P 500: A Comparison of Statistical and Machine Learning Approaches (Pending)

## Data Source

The analyses use the **SPDR S&P 500 ETF Trust (SPY)** as a proxy for the S&P 500.

* **Source:** Yahoo Finance
* **Data:** Daily adjusted closing prices
* **Primary sample:** January 1996 – December 2025
* **Return measure:** Daily logarithmic returns

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

This research focuses exclusively on SPY as a proxy for the S&P 500, so the findings may not fully generalize to individual stocks, international markets, or other asset classes. The results also depend on historical data and on modeling choices such as estimation-window length, confidence level, distributional assumptions, and model specification. Because financial markets evolve over time, structural changes, volatility regime shifts, and rare extreme events may reduce the ability of historical relationships to represent future risk.

The analysis was designed to avoid look-ahead bias by using only information available before each forecast date. However, researcher choices regarding model selection, parameter settings, evaluation periods, and performance metrics can still introduce selection bias. In addition, passing or failing a statistical backtest does not by itself prove that a model accurately represents the true market-risk process, particularly because extreme tail events are relatively rare and therefore provide limited observations for evaluation.

