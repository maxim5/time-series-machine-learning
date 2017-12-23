# Time Series Prediction with Machine Learning

A collection of different Machine Learning models predicting the time series, 
concretely the market price for given the currency chart.

# Requirements

Required dependency: `numpy`. Other dependencies are optional, but to diversify the final models ensemble, 
it's recommended to install these packages:  `tensorflow`, `xgboost`.
Tested with python version: 2.7

# Fetching data

There is one built-in data provider, which fetches the data from [Poloniex exchange](https://poloniex.com/exchange).
Currently, all models have been tested with crypto-currencies' charts.

Fetched data format is standard security [OHLO trading info](https://en.wikipedia.org/wiki/Open-high-low-close_chart): 
date, high, low, open, close, volume, quoteVolume, weightedAverage.
But the models are agnostic of the particular time series features.

Unless explicitly specified, all scripts work with following tickers: 
`BTC_ETH`, `BTC_LTC`, `BTC_XRP`, `BTC_DGB`, `BTC_STR`, `BTC_ZEC`.

To fetch the data, run `run_fetch.py` script from the root directory:

```sh
# Fetches the default tickers.
./run_fetch.py
```

By default, the data is fetched for all time periods available in Poloniex (day, 4h, 2h, 30m, 15m, 5m) 
and is stored in `_data` directory. One can specify the tickers via command-line arguments.

```sh
# Fetches just BTC_ETH ticker data.
./run_fetch.py BTC_ETH
```

**Note**: the second and following runs *won't* fetch all data from scratch, but just the update from the last run till now.

# Training the models

```sh
# Trains all models until stopped.
./run_train.py
```

By default, the script trains all available models (see below) with random hyper-parameters, cross-validates each model and
saves the result weights if the performance is better than current average. All models are placed to the `_zoo` directory
(note: it is possible that early saved models will perform much worse than later ones, so 
you're always welcome to clean-up the models you're definitely not interested in).

Each model has the following run parameters:
 - ticker name, e.g., `BTC_ETH`
 - time period, e.g. `4h`
 - target column, e.g. `high` (means the model is predicting the next high)

Currently supported methods:
- Ordinary linear model
- Gradient boosting (using `xgboost` implementation)
- Feed-forward neural network (in `tensorflow`)
- Recurrent neural network: LSTM, GRU, one or multi-layered (in `tensorflow` as well)

# Running predictions

```sh
# Runs all models for BTC_ETH ticker and outputs the aggregated prediction.
./run_predict.py BTC_ETH
```

Downloads the current trading info for the selected currencies and runs all models that 
have been saved for these currencies and time period.

# License

Apache 2.0
