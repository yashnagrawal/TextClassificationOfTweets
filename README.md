# Sentiment Analysis with Multinomial Naive Bayes
This code is designed to perform sentiment analysis using a Multinomial Naive Bayes classifier on a dataset of tweets. It preprocesses the data, trains the classifier, evaluates its performance, and displays a confusion matrix.

# Getting Started
To get started, make sure you have the required libraries installed:

- numpy
- pandas
- matplotlib
- seaborn
- scikit-learn
- nltk
You can install these libraries using pip:

Copy code
```bash
pip install numpy pandas matplotlib seaborn scikit-learn nltk
```

# Usage
Data Preparation: Place your training data in a CSV file with columns 'id', 'label', and 'tweet' representing tweet IDs, sentiment labels, and tweet text, respectively.

Code Execution: Run the Python script sentiment_analysis.py. The script will perform the following steps:

1. Load and preprocess the dataset.
2. Split the dataset into training and testing sets.
3. Clean and vectorize the text data.
4. Train a Multinomial Naive Bayes classifier.
5. Evaluate the classifier's performance.
6. Display a confusion matrix.
7. Results: The script will output the accuracy and classification report, including precision, recall, F1-score, and support for each class (Non-Hate and Hate).

# Customization
You can customize the code by modifying the following:

- Input data file path: Change the file_path variable to specify the path to your CSV file.

- Classifier settings: You can adjust the classifier settings, such as the alpha parameter for smoothing, in the train_mnb_model function.

- Text preprocessing: If you want to customize the text preprocessing steps, you can do so in the clean_tweet function.

