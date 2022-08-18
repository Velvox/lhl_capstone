# Netflix Stock Prediction Enhanced with Sentiment.
This project aims to enhance time series predictions of the Netflix stock price with news and twitter sentiment. Using these new features we will aim to improve a baseline model's performance and prove out the use of sentiment to predict stock price.

Project slide deck:
https://docs.google.com/presentation/d/1IpHQJVaRVdgiPJQiatTl1VZGb1A3GhGJKNvoHarub-w/edit?usp=sharing

## Deadline
- August 18th

## Project Goal
Given information we have today, learn what features improve stock price prediction of tomorrow, considering online sentiment.

## Key Learnings
#### Adding sentiment to the model improved prediction acuraccy measurably, when used in a 14-day proof-of concept.

![Sentiment Feature Improvements](/imgs/prediction_results.png)

#### News sentiment was a much more useful feature in prediction, when compared to Twitter

![Feature Importances](/imgs/feature_importances.png)

#### Parsing through Twitter will require more work, due to the volume of non-relevant tweets

![Topics](/imgs/lda.png)

## Project steps
- [x] Gather data on stock price, tweets and news text about Netflix
- [x] Build a baseline model to be improved upon.
- [x] Classify sentiment using FinBERT / Vader
- [x] Make new predictions with sentiment, identify improvement, if any.
- [x] Stretch: Perform topic analysis, informed by sentiment
- [ ] Stretch: Deploy the model on Flask

## Workflow

![Workflow](/imgs/tech_stack.png)

## Recommendations
- Collect 3 months to a year of Twitter and news data to build a more effective model across multiple companies.
- Filter news & twitter content with keywords identified in LDA topic analysis. Focus on news data moving forward.
- Deploy the model and use online learning to make daily optimizations and predictions.
- With more data, start multi-step time series predictions with sentiment.
