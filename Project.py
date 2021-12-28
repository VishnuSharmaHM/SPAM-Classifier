import nltk
from sklearn.naive_bayes import MultinomialNB
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score

df1 = pd.read_csv("data_train.csv", encoding='latin-1')
df1 = df1[df1['text'].notna()]
df1['spam'] = df1['spam'].apply(lambda x: 0 if x == False else 1)
df1 = df1.loc[:, ~df1.columns.str.contains('^Unnamed')]

df2 = pd.read_csv("data_train_1.csv", encoding='latin-1')
df2 = df2[df2['text'].notna()]
df2['spam'] = df2['spam'].apply(lambda x: 0 if x == "ham" else 1)
df2 = df2.loc[:, ~df2.columns.str.contains('^Unnamed')]

df3 = pd.read_csv("data_train_2.csv", encoding='latin-1')
df3 = df3[df3['text'].notna()]
df3['spam'] = df3['spam'].apply(lambda x: 1 if x =="spam" else 0)
df3 = df3.loc[:, ~df3.columns.str.contains('^Unnamed')]

df=pd.concat([df1,df2,df3],axis=0)

x_train, x_test, y_train, y_test = train_test_split(df['text'], df['spam'], test_size = 0.2, random_state = 0)
lst = x_train.tolist()
vectorizer = TfidfVectorizer(input= lst ,lowercase=True,stop_words='english')
features_train_transformed = vectorizer.fit_transform(x_train)
features_test_transformed  = vectorizer.transform(x_test)
classifier = MultinomialNB()
classifier.fit(features_train_transformed, y_train)
predicted = classifier.predict(features_test_transformed)
actual = y_test.tolist()
results = confusion_matrix(actual, predicted)
print('Confusion Matrix for Train Data:')
print(results)
print ('Accuracy Score of train data %:  ',accuracy_score(actual, predicted)*100)

df4 = pd.read_csv("data_test.csv", encoding='latin-1')
df4['spam'] = df4['spam'].apply(lambda x: 1 if x =="spam" else 0)
X_test = vectorizer.transform(df4['text'])
pred=classifier.predict(X_test)
actual=df4['spam']
results = confusion_matrix(actual, pred)
print('Confusion Matrix for Test Data:')
print(results)
print ('Accuracy Score of test data % :  ',accuracy_score(actual, pred)*100)

file1 = open("predictions.csv", "w")
count=0
file1.write(str("Output"))
file1.write("\n")
for i in pred:
    count+=1
    if(i==0):
        file1.write(str("Ham"))
        file1.write("\n")
    elif(i==1):
        file1.write(str("Spam"))
        file1.write("\n")
file1.close()
print("Number of value in predictions.csv file -",count)