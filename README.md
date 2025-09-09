# SENTIMENT-ANALYSIS-WITH-NLP

COMPANY :  CODTECH IT SOLUTIONS

NAME :  POOJA SHREE M I

INTERN ID  :  CT04DY1390

DOMAIN  :  MACHINE LEARNING

DURATION  :  4 WEEKS

MENTOR  :  NEELA  SANTHOSH

**Sentiment Analysis is one of the most popular applications of Natural Language Processing (NLP), where the main objective is to automatically determine whether a piece of text expresses a positive, negative, or neutral sentiment. This task has wide applications in business, customer support, product reviews, social media monitoring, and more. In this project, we implemented a sentiment analysis model using TF-IDF (Term Frequency–Inverse Document Frequency) vectorization for text feature extraction and Logistic Regression as the classification algorithm.

The first step in this task was to collect or create a dataset of customer reviews along with their corresponding sentiment labels. Each review sentence was assigned a sentiment category such as positive, negative, or neutral. Since raw text data cannot be directly used by machine learning algorithms, it was important to preprocess the text. Preprocessing involved several steps such as converting all text to lowercase, removing unwanted characters like punctuation, links, and special symbols, and removing common stopwords such as “the”, “is”, and “and”. However, for sentiment analysis, words like “not” or “no” play an important role in determining polarity, so they were retained in the cleaned data.

After preprocessing, the text was converted into a numerical representation using TF-IDF vectorization. TF-IDF is an effective method for representing text in machine learning tasks. It considers not only the frequency of words in a document but also how important those words are across the entire dataset. For example, common words like “good” or “bad” may appear in many documents, while more unique terms have higher importance. This method helps the model understand which words are significant in distinguishing sentiment categories.

The dataset was then divided into training and testing sets to evaluate the performance of the model. The training set was used to train a Logistic Regression classifier. Logistic Regression is a widely used algorithm for classification problems, and it works well with high-dimensional data like TF-IDF vectors. It learns to find the best decision boundary that separates positive, negative, and neutral reviews based on the patterns in the data.

Once the model was trained, predictions were made on the test dataset. The performance was measured using evaluation metrics such as accuracy, precision, recall, F1-score, and a confusion matrix. The confusion matrix helped visualize how many reviews were correctly classified and where the model made mistakes. Additionally, sample test sentences were given to the model to check if it correctly identified their sentiments. For example, a sentence like “The product was not good” was predicted as negative, which demonstrates that the model successfully captured sentiment even with negations.

The results showed that the combination of TF-IDF and Logistic Regression provides a simple yet effective solution for sentiment analysis tasks. Although deep learning models such as LSTMs or transformers can achieve higher accuracy on large datasets, TF-IDF with Logistic Regression is lightweight, fast, and interpretable, making it suitable for small to medium-sized projects.

In conclusion, this task provided practical experience in implementing NLP techniques, preprocessing textual data, applying feature extraction, training a machine learning model, and evaluating performance. Sentiment analysis is a valuable tool for businesses to understand customer opinions, improve products, and enhance user satisfaction.**
