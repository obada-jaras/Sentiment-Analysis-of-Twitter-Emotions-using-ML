# **Sentiment Analysis of Twitter Emotions using ML**

This project code implements sentiment analysis on Twitter data using machine learning algorithms to classify tweets as positive or negative. The code utilizes five different algorithms including Random Forest, Naive Bayes, K-Nearest Neighbors, Multi-Layer Perceptron (MLP), and XGBoost. The input data consists of two files, `PositiveTweets.tsv` and `NegativeTweets.tsv`, containing positive and negative tweets respectively. The tweets are preprocessed to remove stop words, non-Arabic characters, and consecutive redundant characters, and also undergo text normalization through stemming. In addition to this, various features such as content length, number of tokens, number of hashtags, number of bad words, and number of emojis are added to the data to enhance the analysis. The preprocessed data along with the added features is then used to train and evaluate the performance of the machine learning models.

<br>
<hr>
<br>

## **Table of Contents**
- [**Overview**](#overview)
  - [**Data**](#data)
  - [**Features Extraction**](#features-extraction)
  - [**Pre-processing Data**](#pre-processing-data)
  - [**Machine Learning Models**](#machine-learning-models)
  - [**Evaluation**](#evaluation)
- [**Prerequisites**](#prerequisites)
- [**Usage**](#usage)

<br>
<hr>
<br>

## **Overview**

### **<u>Data</u>**
The code reads two files, `PositiveTweets.tsv` and `NegativeTweets.tsv`. These files contain positive and negative Arabic tweets respectively. The positive tweets file contains <u>23879</u> tweets and the negative tweets file contains <u>23122</u> tweets. The tweets are then preprocessed and used to train and evaluate the machine learning models. The data is balanced, so the positive and negative tweets are approximately equal in number, allowing the models to make accurate predictions without being biased towards one sentiment or the other.

<br>
<hr>
<br>

### **<u>Features Extraction</u>**

The code adds several features to the data to enhance its performance. These features are:

- Content Length (before and after [pre-processing](#pre-processing-data)): The number of characters in each tweet.
- Number of Tokens (before and after [pre-processing](#pre-processing-data)): The number of words in each tweet.
- Number of sentences before [pre-processing](#pre-processing-data).
- Number of Hashtags: The number of hashtags used in each tweet.
- Number of Bad Words: The number of words in each tweet that are considered bad words.
- Emojis Features:
  - Number of _Love Emojis_.
  - Number of _Broken Heart Emojis_.
  - Number of _Happy Emojis_.
  - Number of _Smile Emojis_.
  - Number of _Sad Emojis_.
  - Number of _Angry Emojis_.
  - Number of _Surprising Emojis_.
  - Number of _Thinking Emojis_.
  - Number of _Flowers Emojis_.
  - Number of _Moon and Sun Emojis_.
  - Number of _Hands Emojis_.
  - Number of _Prohibiting Emojis_.
- Words Features:
  - Number of times the words `يارب` and `يا رب` appear in the tweet.
  - Number of times the word `لله` appears in the tweet.
  - Number of times the word `الحمد` appears in the tweet.
  - Number of times positive words such as `جميل`, `جمال`, `حب`, `خير`, and `صباح` appear in the tweet.
  - Number of times negative words such as `عيب`, `غلط`, `تعب`, `كئيب`, `قرف`, `مرض`, `موت`, `سيء`, `مشكل`, `خرا`, `زفت`, `ظلم`, and `كذب` appear in the tweet.

<br>
<hr>
<br>

### **<u>Pre-processing Data</u>**

After preprocessing, additional features that depend on the preprocessed text are added (content length and number of tokens). This improves the models and increases their accuracy in classifying the tweets. The preprocessing phase is:

1. Removing Stop Words: Stop words are common words that do not contain important meaning and are usually removed from the texts. In our code, we remove Arabic stop words such as "من", "الى", "عن", and "في".
2. Removing Non-Arabic Characters: Any non-Arabic characters are removed from the tweets as they do not contribute to the sentiment analysis.
3. Removing Consecutive Redundant Characters: Consecutive redundant characters are removed, for example, "انااااااااا" and "اناااا" would become "انا".
4. Stemming: Stemming is the process of reducing words to their root form. In our code, we use [ARLSTem](https://github.com/nltk/nltk), an Arabic stemmer, to stem the words in the tweets.

<br>
<hr>
<br>

### **<u>Machine Learning Models</u>**

1. **Random Forest**: A random forest is an ensemble learning method for classification, regression, and other tasks, that operate by constructing a multitude of decision trees at training time and outputting the class that is the mode of the classes (classification) or mean prediction (regression) of the individual trees.
   
2. **XGBoost**: XGBoost is an implementation of gradient boosting trees. It uses decision trees as the weak learners and ensembles many of them to produce a more robust model. XGBoost is known for its speed and performance, especially when working with large datasets.
   
3. **Multi-Layer Perceptron (MLP)**: MLP is a feedforward artificial <u>neural network</u> model that maps sets of input data onto a set of appropriate outputs. It consists of multiple layers of nodes connected through weighted connections.
   
4. **K-Nearest Neighbors (KNN)**: KNN is a non-parametric, lazy learning algorithm. It can be used for both classification and regression problems. The algorithm stores all available cases and classifies new cases based on a similarity measure.
   
5. **Naive Bayes**: Naive Bayes is a probabilistic algorithm based on Bayes' theorem with the assumption of independence between predictors. The algorithm is particularly suited when the dimensionality of the inputs is high.

<br>
<hr>
<br>

### **<u>Evaluation</u>**

To evaluate the performance of the models, we use several evaluation metrics, including:
1. Accuracy: It measures the ratio of correct predictions to the total number of predictions.
2. Precision: It measures the number of true positive predictions relative to the total number of positive predictions.
3. Recall: It measures the number of true positive predictions relative to the total number of actual positive cases.
4. F1 Score: It is a weighted average of precision and recall, where the best value is 1 and the worst value is 0.
5. Confusion Matrix: It is a table used to evaluate the performance of a classification algorithm, showing the true positive, false positive, false negative, and true negative predictions.

<br>


| Model | Execution Time | Accuracy | Precision | Recall | F-measure |
|-------|----------------|----------|----------|--------|----------|
| Decision Tree | 3.8s | 0.8857 | 0.8857 | 0.8857 | 0.8856 |
| XGBoost | 2.1s | 0.8635 | 0.8635 | 0.8635 | 0.8635 |
| MLP | 5m 9.1s | 0.8614 | 0.8615 | 0.8614 | 0.8615 |
| KNN | 9.1s | 0.7964 | 0.7969 | 0.7964 | 0.7964 |
| Naive Bayes | 0.5s | 0.7836 | 0.8279 | 0.7836 | 0.7748 |

<br>
<hr>
<hr>
<br>

## **Prerequisites**

Before getting started with the project, you will need to have the following software installed on your machine:

1. [Python 3](https://www.python.org/downloads/).
2. Required Libraries: You will need to install <u>numpy</u>, <u>pandas</u>, <u>emoji</u>, <u>nltk</u>, <u>xgboost</u>, and <u>scikit-learn</u> by running `pip install numpy pandas emoji nltk xgboost scikit-learn` on the command line.

<br>
<hr>
<hr>
<br>

## **Usage**

To run the code for this project, make sure you have the required libraries installed. Check the [Prerequisites](#prerequisites) section for more information on the libraries.

There are two options for running the code: using a Python script or using a Jupyter Notebook. You can choose whichever option is more convenient for you. The code and the data files are included in this project.
