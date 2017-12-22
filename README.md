# Time Series Prediction with Machine Learning

A collection of different Machine Learning models trying to predict the time series, concretely the market price for given the currency chart.
Currently supported:
- Ordinary linear model
- Gradient boosting (using `xgboost` implementation)
- Feed-forward neural network (in `tensorflow`)
- Long-short term memory neural network (LSTM, in `tensorflow` as well)

# Requirements

Required dependency: `numpy`. Other dependencies are optional, but to diversify the final models ensemble, 
it's recommended to install these packages:  `tensorflow`, `xgboost`.
Tested with python version: 2.7

# Fetching data

There is one built-in data provider, which fetches the data from [Poloniex exchange](https://poloniex.com/exchange).
Currently, all models have been tested with crypto-currencies' charts.

Fetched data format is standard security [OHLO trading info](https://en.wikipedia.org/wiki/Open-high-low-close_chart): date, high, low, open, close, volume, quoteVolume, weightedAverage.
But the models are agnostic of the particular time series features.

To fetch the data, run from the root directory:

```sh
# By default, downloads BTC_ETH, BTC_DGB, BTC_STR, BTC_ZEC, for all time periods.
# Target directory: `_data`
./run_fetch.py
```

# Training the model

```sh
# Trains all models until stopped.
./run_train.py
```

By default, the script trains all available models with random hyper-parameters, cross-validates each model and
saves the result model if it's better than current average. All models are placed to the `_zoo` directory.

It is possible that early saved models will be much worse than later ones, you're always welcome to clean-up the models
you're definitely not interested in.

# Running predictions

```sh
./run_predict.py
```

By default, downloads the current trading info for selected currencies and runs all models that 
have been saved for this currency and time period.

# License

Apache 2.0
