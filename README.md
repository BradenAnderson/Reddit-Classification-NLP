# Subreddit Prediction

### Project Goal: Given a post from either the wallstreetbets or CryptoCurrency subreddit, predict which subreddit that post came from.


**Note:** The datasets for this project can be downloaded [from my google drive here](https://drive.google.com/drive/folders/1qKqFijuDcDpiewRdy_WybI73n7l2-F1k?usp=sharing)


**Project Summary:** 

It is no suprise that data hungry deep learning models have become increasingly popular in recent years, as the amount and availability of data to train them with is now greater than ever. In fact, when researching the benefit of deep learning models, you may come across a wide variety of charts [that look like these](https://www.google.com/search?q=model+performance+vs+amount+of+training+data+deep+learning&sxsrf=ALeKk01qChuSaLZZQW6VYlV8VOnkam7dyA:1625542257821&source=lnms&tbm=isch&sa=X&ved=2ahUKEwis0ODmwM3xAhUBvp4KHWkmCq8Q_AUoAXoECAEQAw&biw=1463&bih=741#imgrc=9krMgpgfVyr0_M) reinforcing this idea that deep learning models tend to outperform other approaches when there is a large amount of data available to train on. In this project, I decided to try and verify this for myself in the context of a natural language processing classification problem. To accomplish this, I first collected approximately 1 million posts each from the wallstreetbets and CryptoCurrency subreddits. After preprocessing the data, I noticed a large variation in the size of a given reddit post, which ranged from 1 to several thousand words. I decided to limit the problems scope to only include posts that were between 5 and 20 words in length, which left me with around 1.1 million posts in total. Of these 1.1 million posts I immediately set aside 50,000 of them - 25,000 for validation (used during neural network training) and another 25,000 for a test set which I used as the benchmark to compare model performances. The remainder of the posts were available to pull from when creating  training sets. I then selected four models, two classic machine learning approaches (Naive Bayes and Gradient Boosted Decision Trees) and two variants of a bidirectional recurrent neural network (one utilizing pretrained fasttext word vectors in the embedding layer, and the other which learned the word embeddings during network training). For the classic ML models, an initial set of gridsearches were performed with a 5000 sample subset of the training data. These gridsearches were used to find hyperparameter settings which were then held constant throughout the rest of the investigation. I then trained each model using several different training set sizes ranging from 100 to 500,000 samples and each time evaluated their performance on the test set. The results of this analysis showed that the deep learning approaches did consistently out perform the classic ML models once the training set exceeded a few thousand samples. I also somewhat suprisingly found that all models were able to achieve reasonably high accuracy (90% or greater), which I believe is the result of the rigorous text preprocessing I performed as well as the decision to further standardize the data by limiting to posts between 5 and 20 words in length. Another suprising observation is that no model seemed to significantly beneifit from a larger training set after about 15k samples. Further error analysis would need to be performed in order to confirm this, but my assumption is that the incorrectly classified samples in the test set may be some rare or anomolous examples that are not well represented by the rest of the available data. In conclusion, the results of the analysis I performed in this project do support the claims that deep learning is likely to outperform other approaches when there is a large amount of training data available. However these results fall short of recreating the model performance vs training set size charts I linked to at the start of this summary. My recommendations for further exploration would be to reperform this analysis using a much larger test set that may provide a more diverse set of challenging examples to evaluate each model with. Additionally, it may be worth revisiting the decision to limit the data to posts of a certain length and instead reperform the analysis with a greater variety of posts included. 

***

### Table of Contents

[Data Collection](#data-collection)

[Text Preprocessing](#text-preprocessing)

[Exploratory Data Analysis](#exploratory-data-analysis) 

[Creating Datasets](#creating-datasets) 

[Model Evaluation](#model-evaluation) 

[Deep Learning](#deep-learning) 

[Analysis of Model Performance](#analysis-of-model-performance) 

***
***


### Data Collection

**Notebook:**

00_Data_Collection.ipynb


**Description:**

The purpose of the 00_Data_Collection notebook is to efficiently utilize the pushshift.io Reddit API to collect text data from any desired subreddit(s). To accomplish this goal, several functions were constructed that automate the process of using this API to make requests for data and save the results to .csv file(s). Specifically, using the scrape_multiple_subreddits function a user can obtain any desired number of posts from any list of subreddit pages with a single function call. The progress of the script can be easily tracked through the helpful print outs it provides. Additionally, check point save files will be generated at any user defined interval which allows the data to start being reviewed and utilized prior to completing the collection process. The flexibility of this function allows you to set up and exucute it one time and then step away while the script continues to run in the background, collecting the exact information you requested without requiring any additional input. This function is also robust to intermittent website connection issues. This means that instead of crashing and requiring user intervention when a request for reddit data returns an error code, the function will gracefully enter a reconnection loop where it will retry to collect the information every 5 seconds until multiple failures in a row occur or the data is successfully collected. 

**References**

1. pushshift api: [Pushshift Reddit API](https://github.com/pushshift/api)


***

### Text Preprocessing

**Notebooks:**

01_Data_Cleaning.ipynb

02_Data_Cleaning_With_spaCy.ipynb


**Description:**

A significant contributor to the success of any Natural Language Processing project is the ability to create a limited size and information rich vocabulary for your models to use when learning the relationship between the input text and your target variable. With this in mind, I spent a significant amount of effort towards constructing a comprehensive set of preprocessing operations that will result in such a vocabulary. The 01_Data_Cleaning notebook implements the first set of preprocessing operations, which includes the following:

1. Combining the title and selftext columns returned during the reddit scrape into a single all_text_data feature.

2. Replace all emojis with a custom string that describes its meaning.

3. Replace all instances of sms speak (i.e. internet slang) with a standardized text that retains the overall meaning. 

4. Convert all text to lowercase.

5. Replace all contractions with their expanded forms.

6. Remove all punctuation marks.

7. Remove excessively long words (25 characters or more). Note: This step was added during a review of the preprocessed output text, when I realized that the reddit text data occasionally included websites and other long strings that were not actual words. Since I believed that each specific website was likely rare and unlikely to contain much useful information to the model, I decided that the vocabulary size decrease that resulted from removing them would benefit the model more than the information provided by keeping them in. 

8. Remove any symbols left that are not on the english keyboard (defined as a value greater than 127 when using the python ord() function on the associated character). 

After the preliminary preprocessing steps shown above, the 02_Data_Cleaning_With_spaCy notebook was used to perform lemmatization and stop word removal. The lemmatizer I used is built into the en_core_web_lg pretrained model pipeline, for more information on this model please see the references below.

**References**

1. spaCy pretrained english language models: [spaCy english models](https://spacy.io/models/en)
2. wikipedia article explaining sms speak: [Wikipedia SMS Speak](https://en.wikipedia.org/wiki/SMS_language)

***

### Exploratory Data Analysis

**Notebook:**

03_EDA.ipynb

**Description:**

The purpose of the exploratory data analysis notebook is to get some insight into, and gain familarity with, the text data prior to starting the modeling process. To accomplish this I chose to create bar charts of the most frequent words as well as word clouds. For each plot I decided to take three separate views, one where all data was included, one with only the wallstreetbets posts included, and another with only the CryptoCurrency posts. This process allowed me to get a good understanding of what words were likely going to be the greatest contributors towards a models ability to distinguish which subreddit a post came from. For example, words like "gme" and "stock" are likely going to be strong indicators that a post came from wallstreet bets, while words like "cryptocurrency" and "bitcoin" will be strong idicators of a r/Cryptocurrency thread. 

This EDA step also helped validate the extra effort I put in to replace emojis with a custom piece of text rather than discard them. Specifically, the terms "rocket" and "moon" are both in the top 10 most frequent words for wallstreetbets posts, meaning they are likely going to large contributors towards a models ability to identify this subreddit. Both of these terms were initialy emojis that I replaced with text during the data cleaning phase. 

**References**

1. This is only tangentially related but I think it's interesting in any case. Here is Andreas Muellers (Microsoft SDE and core Scikit-Learn developer) blog where he talks about how he built the wordcloud library I used. [Word Cloud Blog Post](https://peekaboo-vision.blogspot.com/2012/11/a-wordcloud-in-python.html)

***

### Creating Datasets

**Notebook:**

04_Incrementing_Train_Set_Size.ipynb



**Description:**

This notebook contains all of the code needed to efficiently create all of the .csv file and tensorflow datasets needed to train the classic machine learning and deep learning models. 


***

### Model Evaluation


**Notebooks:**

05_Gradient_Boosting.ipynb

05_Naive_Bayes.ipynb

06_Evaluate_Models.ipynb


**Description:**

The 05 series notebooks contain the initial 5000 sample gridsearches that were used to determine the hyperparamters for the Gradient Boosted Decision Tree and Naive Bayes models which were then held constant throughout the remainder of the analysis. The 06_Evaluate_Models notebook retrains the best models found in the 05 series notebooks using a wide range of different training set sizes. After each round of training the model is evaluated on the 25,000 sample test set and the models performance is recorded in a file that will be referenced in the final model performance comparison notebook. 

***

### Deep Learning

**Notebook:**

07_LSTM.ipynb


**Description:**

The 07_LSTM notebook contains all of the code needed to efficiently train a bidirectional recurrent neural network using a list of training set sizes and record each models performance on the test set. The neural network is constructed inside a function that has parameters which allow the user to decide whether the neural network should be trained using an embedding layer that is populated with pretrained fasttext word vectors or a trainable embedding layer that will learn word vector representations simultenously with the LSTM and Dense layer weights. To facilitate the option to populate the embedding layer with fasttext vectors, the start of the notebook includes a set of functions which train a fasttext word vector model on the set of all available training data. This fasttext model is then saved and reloaded any time there is a need to populate an embedding matrix with word vectors. 

**References**

1. [fasttext](https://fasttext.cc/docs/en/support.html)
2. Deep Learning with Python by Francois Chollet.


***

### Analysis of Model Performance

**Notebook:**

08_Model_Performance.ipynb



**Description:**

This notebook loads the results of the 06_Evaluate_Models and 07_LSTM notebooks and makes several charts that show the accuracy vs training set size for the various models. 

***
