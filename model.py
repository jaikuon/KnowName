#data is from https://www.kaggle.com/datasets/amaleshvemula7/name-and-country-of-origin-dataset
import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB, ComplementNB
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score, classification_report

# Function to load and preprocess data
def loaddata(file_path):
    df = pd.read_csv(file_path)
    df = df.dropna(subset=['Name', 'Country'])
    return df

# Function to train and save the model
def trainmodel(df, model_path='pipeCNB_model.joblib'):
    x = df['Name']
    y = df['Country']
    X_train, X_test, Y_train, Y_test = train_test_split(x, y, test_size=0.2, random_state=22)

    pipeCNB = Pipeline([('tfidf', TfidfVectorizer()), ('clf', ComplementNB())])
    pipeCNB.fit(X_train, Y_train)

    joblib.dump(pipeCNB, model_path)

    predictCNB = pipeCNB.predict(X_test)
    print(f"ComplementNB Accuracy: {accuracy_score(Y_test, predictCNB)}")
    print(classification_report(Y_test, predictCNB))

# Function to load the model and make a prediction
def usemodel(name, model_path='pipeCNB_model.joblib'):
    pipeCNB = joblib.load(model_path)
    prediction = pipeCNB.predict([name])
    return prediction

# Main block to train and save model
if __name__ == "__main__":
    dataset_path = 'final_all_names_code.csv'
    df = loaddata(dataset_path)
    trainmodel(df)

