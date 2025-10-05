# Stock Movement Predictor

An ML-driven application that predicts significant stock price movements and sends push notifications. API available at https://stock-notifier-api.fly.dev/docs. To run, create an account with the `/auth/register` endpoint, copy the access token and enter that in the authorize button at the top of the swagger ui. To login, use the `/auth/token` endpoint. Other notable endpoints are the `/user/stocks` endpoint to add stocks to your profile, `/user/settings` to tweak your prediction needs, and `/tune` to train the model witht he best hyperparam configuration. Finally, `/predict` for new predictions on your saved stocks.

## Project Overview

The purpose of this project is to evaluate the effectiveness of using technical indicators as a predictive measure for a stock's future price. Additionally, it is an exercise in building a scalable software system which incorporates user customization and stock notifications. To accomplish these things I use a Random Forest classifier based on indicators such as RSI, bolling bands, MACD, and Volume. Instead of a typical regression task predicting a specific stock price, I am trying a binary classifier to predict a percentage movement over a user defined timeframe. For example, the user set a notification if with 90% certainty, a stock will move 3% in the 24 hours when trained on a daily or hourly timeframe. I employ a variety of technologies to make and push out predictions to users, such as Google Cloud Storage, Alpha Vantage/Yahoo APIs, Redis for caching stock data and pub/sub with notifications, AWS Lambda for triggering prediction notifications, telegram APIs for notifications themselves, and fly.io for quick microservice deployment.

## Feature Selection, Data, and Model

A big problem when training stock data is that the range of values for a technical indicator like bollinger bands or MACD greatly differs depending on the price so training multiple stocks into the same model or over many years may not be effective. To attempt to alleviate this, I use one model per stock for a user, and on the technical indicator side for bollinger bands I am going to calculate %b to reduce price dependency.

```math
\%B = \frac{\text{Current Price} - \text{Lower Band}}{\text{Upper Band} - \text{Lower Band}}
```

For MACD, I am doing the MACD Histogram which factors the signal.

```math
\text{MACD Line} = \text{EMA}_{12} - \text{EMA}_{26}
```
```math
\text{Signal Line} = \text{EMA}_{9}(\text{MACD Line})
```
```math
\text{MACD Histogram} = \text{MACD Line} - \text{Signal Line}
```

RSI is already a decent predictor of overbought/oversold conditions with values ranging from 0 to 100. Finally, I will use volume. These features should provide good numbers for thresholding that occurs in the decision trees that are ensembled in Random Forests. For model selection I will use a Random Forest Classifier and fine tune the `n_estimators`, `max_depth`, and `max_features` hyperparameters with  Random Search in an attempt to get the most out of the model. Another problem I have with a lot of stocks is a very imbalanced data set with a large portion of the data being the positive movement class for small percentage thresholds and vice versa for large percentage thresholds. For example, a 4% movement over a 12 hour period will be very rare. An example of this, Tesla over a 15 year period is 88%/12% for movement above the 3% threshold. To counter this, proper stock threshold configurations is important for maximizing potential gains, and I will consider using SMOTE to artificailly generate the less frequent class. Additionally, stratified sampling may be useful, but in reality as described below I do not end up using SMOTE or stratified because I opt out of typical cv fold train/test split setup to avoid data leakage with the time series based data.

## Model Performance

The main training and performance evaluation happens in the `/tune` endpoint. I use `RandomizedSearchCV` to fine tune the RF hyperparameters and found that these values below generally give us the best performance with some variance depending on the stock:

| n_estimators   | max_features  | max_depth  |
|------------|------------|------------|
| 221 | None (all features) | 9 |

For the cross validation I use 3 folds and a 20% split for the final holdout set. Using the hourly or daily timeframe leads to better performance as there is many more data points available to train on than larger frames like the weekly. Certain stocks like Tesla have fairly even class distributions with the proper threshold filters, but others lean heavily to one side especially when looking at larger movement thresholds so for this reason I played with stratified sampling in the random search but had to move away from `StratifiedKFold` which can be found in previous versions on github to use a `TimeSeriesSplit` with gaps the size of the prediction window to prevent data leakage and giving a false sense of hope with the validation/training metrics and poor performance when the model predeicts in the wild. For the even class distributions I played with `roc_auc` as the evaluation metric and `average_precision` which acts as the precision recall AUC for the uneven distributions. Generally, I found about 70-80% roc-auc/pr-auc scores on the validation sets and final holdout set with similar performance on the stratified sets, and while this is certainly better than random chance, this is likely due to data leakage. Typically the best features were MACD histogram, volume, and rsi with %b a bit behind. Below is an example set of data returned from the `/tune` call on Tesla with the `TimeSeriesSplit` and we find the model performs no better than random. This is a disappointing outcome with technical indicators as a predictive measure.

```json
{
  "status": "success",
  "message": "Tuning completed for TSLA",
  "symbol": "TSLA",
  "tuning": {
    "best_params": {
      "n_estimators": 221,
      "max_features": null,
      "max_depth": 9
    },
    "best_cv_score": 0.5128175348111046,
    "cv_folds": 3,
    "n_iter": 20,
    "scoring": "roc_auc"
  },
  "holdout_performance": {
    "accuracy": 52.46,
    "precision": 52.44,
    "recall": 52.46,
    "f1_score": 52.03,
    "roc_auc": 52.52,
    "class_metrics": {
      "no_movement": {
        "precision": 52.33,
        "recall": 42.82,
        "f1_score": 47.1,
        "support": 341
      },
      "significant_movement": {
        "precision": 52.55,
        "recall": 61.89,
        "f1_score": 56.84,
        "support": 349
      }
    },
    "class_distribution": {
      "no_movement": 49.42,
      "significant_movement": 50.58
    },
    "confusion_matrix": {
      "true_negative": 146,
      "false_positive": 195,
      "false_negative": 133,
      "true_positive": 216
    },
    "top_features": {
      "MACD_Hist_Z": 26.82,
      "Volume": 25.73,
      "RSI": 24.79,
      "PercentB": 22.66
    },
    "training_samples": 2764,
    "test_samples": 690
  }
```

## Software System Design
![System Design Diagram](images/stocksoftwarediagram.png)

### User/Model/Prediction service

The meat of the functionality exists in this service hosted at https://stock-notifier-api.fly.dev/docs. The user endpoints could be placed on its own service to reduce any potential tie up with model predictions and frontend user operations. This would be a task if the project ever saw real user load. With the model data, I am pulling stock prices from Alpha Vantage and Redis caching these prices until it is made stale by th etimeframe it is trained on. For example, if it is hourly data, I invalidate it after an hour since fresher data is available. Overall, this massively reduces API calls when doing mass predictions and reduces chances of hitting rate limits. Additionally, I am storing user models in local storage as long as there is available space, which reduces the amount of network load when doing mass predictions and loading/saving models stored in GCS.

### Notification Service

I am using Eventbridge and Lambda to trigger notifications on a schedule. For example, the Eventbridge cron for market open is `cron(30 13 ? * MON-FRI *)`. This works by querying all the users who have notifications enabled at the current time, puts them in a queue (Redis pub/sub), then multiple worker services pick them up, make the predictions and send the telegram message via Telegram's APIs. Additionally, I have a bot service that allows users to link their telegram accounts to their user accounts for this project. Users set a prediction confidence, say 90%, and if the prediction confidence lies above that threshold they get notified, otherwise it passes. 

## Future Improvement

### Sentiment Analysis

The results of the project show that technical indicators alone may not boost odds greater than random chance. It may be helpful to include sentiment analysis on a stock, assign a sentiment score, and include as another feature. For historical sentiment training I would likely need to use a third party that saves this data, but if I want to include it in up to date predictions and notifications this may involve scraping or paying for news feeds to vectorize and determine a postive, negative, neutral, or integer range score (for purposes of a random forest). This would require frequent polling and model retraining as news happens fast and sentiment can change within the hour.

### Vertex AI

I am already using GCP for storing models so it would make sense to use vertex for the prediction service. Batch predictions with vertex would likely simplify the notifications at scale and remove the complexity/bottleneck of the fly.io model predictor service. Better model versioning, while it does not necessarily fit into this project concept may be an option here also.

### Model Retraining

Stock data is constantly updating so user models should be frequently re-trained. A good starting point may be to use a weekly cron job to train a new model, compare the results to the current user's model, and if there is an improvement, replace it. Since Random Forest training and inference with the stock data available is very fast, this task would not be too much of a burden.

### Frontend

I built a small dashboard and authentication UI to add, train, and update notification settings, which is not included in this repo but a standalone tool that I was playing with. It would be nice to build out a more full fledged UI that abstracts the technicality of stock parameters and model training so the user can easily focus on increasing gains. Finally, some stripe subscription functionality was built, but finishing this implementation with customer portal and webhooks could make this a more full fledged tool, which a technically minded customer may want to subscribe to.

![Frontend Image 1](images/6.png)
![Frontend Image 2](images/7.png)
![Frontend Image 3](images/8.png)