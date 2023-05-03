import pandas as pd
import numpy as np
import nltk
import re
from nltk.corpus import stopwords
from nltk import word_tokenize
import string
import matplotlib.pyplot as plt
from wordcloud import WordCloud, STOPWORDS, ImageColorGenerator
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.naive_bayes import BernoulliNB, MultinomialNB
from sklearn.preprocessing import LabelEncoder

df=pd.read_csv(r"C:\Users\User\Desktop\Stress.csv")

df.info()
df.isnull().sum()


nltk.download('stopwords')
stemmer = nltk.SnowballStemmer("english")  # snowball stemmer is Porter stemmer v2 hello
stopword=set(stopwords.words('english'))
def clean(text):                                       # function for cleaning the text column
    text = str(text).lower()
    text = re.sub('\[.*?\]', '', text)
    text = re.sub('https?://\S+|www\.\S+', '', text)
    text = re.sub('<.*?>+', '', text)
    text = re.sub('[%s]' % re.escape(string.punctuation), '', text)
    text = re.sub('\n', '', text)
    text = re.sub('\w*\d\w*', '', text)
    text = [word for word in text.split(' ') if word not in stopword]
    text=" ".join(text)
    text = [stemmer.stem(word) for word in text.split(' ')]
    text=" ".join(text)
    return text
df["text"] = df["text"].apply(clean)

text = " ".join(i for i in df.text)
stopwords = set(STOPWORDS)
wordcloud = WordCloud(stopwords=stopwords,
                      background_color="white").generate(text)
plt.figure( figsize=(15,10))
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis("off")
plt.show()


df["new_label"] = df["label"].map({0: "No Stress", 1: "Stress"})
df = df[["text", "new_label"]]
print(df.head())

x = np.array(df["text"])
y = np.array(df["new_label"])
cv = CountVectorizer()
X = cv.fit_transform(x)
xtrain, xtest, ytrain, ytest = train_test_split(X, y,
                                                test_size=0.33,
                                                random_state=42)

from sklearn.naive_bayes import BernoulliNB
model = BernoulliNB()
model.fit(xtrain, ytrain)
ypred = model.predict(xtest)
print(classification_report(ytest, ypred))


MN = MultinomialNB()
MN.fit(xtrain,ytrain)
print(classification_report(ytest,MN.predict(xtest)))

user = input("Text: I am sad")
data = cv.transform([user]).toarray()
output = model.predict(data)
print(output)
