# Sentiment-Analysis-IMDB-Reviews Dataset

> This project uses Bayes theorem based Naive Bayes classifier to classify the sentiment of IMDB reviews as positive or negative and compares the results with the results obtained by using the scikit-learn library. The Gradio library is used to create a web app for the model prediction.

## Table of Contents

- [Sentiment-Analysis-IMDB-Reviews Dataset](#sentiment-analysis-imdb-reviews-dataset)
  - [Table of Contents](#table-of-contents)
  - [General Information](#general-information)
    - [Algorithm used](#algorithm-used)
    - [Dataset Information](#dataset-information)
  - [Steps involved](#steps-involved)
  - [Result](#result)
  - [Web App](#web-app)
  - [Conclusion](#conclusion)
  - [Technologies Used](#technologies-used)
  - [Acknowledgements](#acknowledgements)
  - [Contact](#contact)
  - [License](#license)

<!-- You can include any other section that is pertinent to your problem -->

## General Information

### Algorithm used

GaussianNB, MultinomialNB, BernoulliNB from Sklearn

### Dataset Information

IMDB dataset having 50K movie reviews for natural language processing or Text analytics.
This is a dataset for binary sentiment classification containing substantially more data than previous benchmark datasets. We provide a set of 25,000 highly polar movie reviews for training and 25,000 for testing. So, predict the number of positive and negative reviews using either classification or deep learning algorithms.
For more dataset information, please go through the following link,
http://ai.stanford.edu/~amaas/data/sentiment/

## Steps involved

- Data Load and Analysis
- Data Wragling
- Exploratory Data Analysis
- Splitting the dataset
- Count Vectorizer transformation
- Modelling
- Model Evaluation and Comparison

## Result

We achieved the following results:

- Multinomial Naive Bayes Classifier
  - Test Accuracy : 0.8563
  - Precision : 0.8697
  - Recall : 0.8393
  - F1 Score : 0.8543
- Precision Recall Curve

![1664216327048](image/README/1664216327048.png)

- ROC Curve

![1664216367122](image/README/1664216367122.png)

- Bernoulli Naive Bayes Classifier
  - Test Accuracy : 0.8474
  - Precision : 0.8724
  - Recall : 0.8152
  - F1 Score : 0.8428
- Precision Recall Curve

![1664216327048](image/README/1664216327048.png)

- ROC Curve

![1664216367122](image/README/1664216367122.png)

## Web App

![1664216683145](image/README/1664216683145.png)

## Conclusion

As it can be observed the Multinomial Naive Bayes Classifier performed better than Bernoulli Naive Bayes Classifier but has higher precision than Multinomial Naive Bayes Classifier. So, we can conclude that Multinomial Naive Bayes Classifier is the best model for this dataset.

## Technologies Used

- Pandas - version 1.3.4
- NumPy - version 1.20.3
- MatplotLib - version 3.4.3
- Seaborn - version 0.11.2
- Scikit-Learn - version 0.24.2

<!-- As the libraries versions keep on changing, it is recommended to mention the version of library used in this project -->

## Acknowledgements

This project was inspired by UpGrad IITB Programme as a case study for the Machine Learning and Artificial Intelligence course.

## Contact

Created by [@sukhijapiyush] - feel free to contact me!

<!-- Optional -->

## License

This project is open source and available without restrictions.

<!-- You don't have to include all sections - just the one's relevant to your project -->
