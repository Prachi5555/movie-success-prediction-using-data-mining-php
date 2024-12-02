#Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.naive_bayes import GaussianNB

#Importing the data set
dataset = pd.read_csv("datasetfinal.csv")

# Split dataset into X (independent variable ) and y (dependent variable)
X = dataset.iloc[:, :-1].values #loc/iloc
y = dataset.iloc[:, 11].values

# Encoding categorical data
# Encoding the Independent Variable
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
number = LabelEncoder()
nameencoder=LabelEncoder()
actor1encoder=LabelEncoder()
actor2encoder=LabelEncoder()
actor3encoder=LabelEncoder()
genresencoder=LabelEncoder()
imdbscoreencoder=LabelEncoder()
budgetencoder=LabelEncoder()
grossencoder=LabelEncoder()
profitencoder =LabelEncoder()

#Encoding each categorical features. 
dataset['director_name'] = nameencoder.fit_transform(dataset['director_name'])
dataset['actor_1_name'] = actor1encoder.fit_transform(dataset['actor_1_name'])
dataset['actor_2_name'] = actor2encoder.fit_transform(dataset['actor_2_name'].astype(str))
dataset['actor_3_name'] = actor3encoder.fit_transform(dataset['actor_3_name'].astype(str))
dataset['genres'] = genresencoder.fit_transform(dataset['genres'])

# dataset['imdb_score'] = imdbscoreencoder.fit_transform(dataset['imdb_score'])
# dataset['budget'] = budgetencoder.fit_transform(dataset['budget'])
# dataset['gross'] = grossencoder.fit_transform(dataset['gross'])
# dataset['profit_percent'] = profitencoder.fit_transform(dataset['profit_percent'])
# Deal with all features. You skipped actor_2_name, actor_3_name. Take care of missing datas in actor_2 and actor_3.
features = ["director_name", "actor_1_name", "genres","imdb_score","budget","gross","profit_percent"]

# Encoding the Dependent Variable
labelencoder_y = LabelEncoder()
y = labelencoder_y.fit_transform(y)

# Splitting the dataset into the Training set and Test set

from sklearn.model_selection import train_test_split 
X_train, X_test, y_train, y_test = train_test_split(dataset[features], y, test_size = 0.2, random_state = 0) 
# test_size = 0.2 means 20% data is in test set
# random_state = 0 means it will generate same test set and train set


from sklearn.preprocessing import StandardScaler
# Scalind the data
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

#Create a Gaussian Classifier
model = GaussianNB()

# Train the model using the training sets
model.fit(X_train, y_train)

# predict values using the training data
nb_predict_train = model.predict(X_test)

# import the performance metrics library
from sklearn import metrics



#Predict Output 
# During training we have given features ["director_name", "actor_1_name", "genres","imdb_score","budget","gross","profit_percent"]
# So to predict provide same fearures in same order. 


actor_name= input("Director Name        : ")
director_name= input("Actor Name           : ")
genre= input("Genre                : ")
imdb_rating= float(input("IMDB Rating          : "))
budget= float(input("Budget               : "))
gross= float(input("Gross                : "))
profit_percent= float(input("Profit Percentage   : "))
# predict=["James Cameron","CCH Pounder","Action|Adventure|Fantasy|Sci-Fi",7.9,237000000, 760505847, 0.031347]
# predict=["Sam Mendes","Christoph Waltz","Action|Adventure|Thriller",6.8,245000000,200074175,-0.183370714]
# predict=["Andrew Stanton","Daryl Sabara", "Action|Adventure|Sci-Fi",6.6,263700000,73058679,-0.722947747]
# predict=["James Cameron","Daryl Sabara","Action|Adventure|Thriller",5.5,26589565,584565,0.032]

predict=[actor_name,director_name,genre,imdb_rating,budget,gross,profit_percent]

predict[0]=nameencoder.transform([predict[0]])
predict[1]=actor1encoder.transform([predict[1]])
predict[2]= genresencoder.transform([predict[2]])
# Since the below ones are numerals no need to label encode them
# predict[3]=imdbscoreencoder.transform([predict[3]])
# predict[4]=budgetencoder.transform([predict[4]])
# predict[5]=grossencoder.transform([predict[5]])
# predict[6]=profitencoder.transform([0.031347])

# Scale or normalize the datas. 
predict = scaler.transform([predict])
prediction = model.predict(predict)


if prediction == 1:
    print("                           HIT")
else:
    print("                           FLOP")
    
          
# Accuracy
print("                             ACCURACY: {0:.4f}".format(metrics.accuracy_score(y_test, nb_predict_train)))
print()
