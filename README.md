# SMS-Fraud-Detection
This project implements an SMS fraud detection system to classify text messages as spam (fraudulent) or ham (legitimate) using machine learning. It includes exploratory data analysis (EDA), model training, and a Streamlit web app for real-time predictions.

# Technologies Used :

- **Python 3.8+** : Core programming language.
- **EDA code for** : To analyze an SMS dataset.
- **Streamlit** : For the web app (app.py).
- **scikit-learn** : For machine learning (Naive Bayes, TF-IDF).
- **pandas** : For data processing.
- **matplotlib, seaborn, wordcloud, nltk** : For EDA visualizations
- **joblib** : For saving/loading the model and vectorizer.
- **VS Code** : Running code


# Dataset

The project uses the UCI SMS Spam Collection dataset by default, which has two columns:
v1: Label (ham or spam).
v2: SMS text.

If using a custom dataset, ensure it has two columns: one for labels (ham/spam or 0/1) and one for text. Update train_model.py and sms_eda.ipynb with the correct column names.



**Download the Dataset** :
- Download the UCI SMS Spam Collection dataset from **Kaggle**.
- Place **spam.csv** in the data/ folder.

- If using a custom dataset, update column names in sms_eda.ipynb and train_model.py (e.g., df = df[['label', 'text']]).
