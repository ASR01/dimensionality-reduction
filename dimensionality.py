import pandas as pd
import numpy as np

from sklearn.preprocessing import LabelEncoder, RobustScaler, MinMaxScaler
from sklearn.decomposition import PCA, TruncatedSVD
from sklearn.model_selection import train_test_split

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import matplotlib.pyplot as plt
import seaborn as sns

#Sonar data()
file = 'sonar.all-data'
#breast cancer data
#file = 'breast-cancer.csv'

#Load the data
df = pd.read_csv('./data/' + str(file))
#df = pd.read_csv('./data/breast-cancer.csv')



#Check the data
print(df.shape,df.head())

# We have 60 columns of features and the last one with the target
#Sonar
print(file[0:5])
if file[0:5] == 'sonar':
	x = df.iloc[:, :-1]
	y = df.iloc[:, -1]
#Breast Cancer
else:
	x = df.iloc[:, 2:-1]
	y = df.iloc[:, 1]



print(x.shape, y.shape)

# We have only two different values so this is a binary classification problem.

labelencoder = LabelEncoder()
y = labelencoder.fit_transform(y)
y = y.reshape([y.shape[0], 1])

print(y.shape)

# Now we scale the features
# We can use the normal scaler or the robustscaler that reduce the noise of values way to deviated of the variance

#x = RobustScaler().fit_transform(x)
x = MinMaxScaler().fit_transform(x)

# Create Training and validation sets

x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.8, random_state=21)

print(x_train.shape, y_train.shape, x_test.shape, y_test.shape)

# Create the model creation sequence



def create_train_model(features, x_train, y_train, x_test, y_test, epochs):
	'''
  	Model building with hyperparameters given by hyp.

  	Args:
    	features = number of features
     	x_train, y_train, x_test, y_test
      	epochs - Number of epochs to train

  	Returns:
    	history - tensorflow trin results
     	model training results
  	'''
    
	model = Sequential()
	model.add(Dense(512,input_dim=features, activation = 'relu'))
	model.add(Dense(64, activation = 'relu'))
	model.add(Dense(1, activation = 'sigmoid'))

	model.compile(optimizer='adam',loss='binary_crossentropy', metrics=['accuracy'])

	history = model.fit(x_train, y_train, epochs=epochs, validation_data = (x_test, y_test))

	return history, model.evaluate(x_test, y_test)



############################## Variables  ##################################


Threshold = 0.9 #Variance kept in dimensionality reduction
epochs = 50
val_split = 0.2

# No dimensionality reductionor
orig_feat = x.shape[1]

history, result = create_train_model(orig_feat, x_train, y_train, x_test, y_test, epochs)
print(result)
print(history.history.keys())

################################ PCA ##########################

# Instantiate PCA without specifying number of components
pca_all = PCA()

# Fit to scaled data
pca_all.fit(x)

# Save cumulative explained variance
cum_var = (np.cumsum(pca_all.explained_variance_ratio_))
n_comp = [i for i in range(1, pca_all.n_components_ + 1)]
dim_PCA = len([i for i in cum_var if i < Threshold])
print(dim_PCA)

# Plot cumulative variance
# ax = sns.pointplot(x=n_comp, y=cum_var)
# ax.set(xlabel='Components', ylabel='Cumulative variance')
#plt.show()

#Generate the PCA with the counted values
pca_3 = PCA(dim_PCA)

# Fit to scaled data
pca_3.fit(x)

# Transform scaled data
x_PCA = pca_3.transform(x)
x_train, x_test, y_train, y_test = train_test_split(x_PCA, y, train_size=0.8, random_state=21)

history_1, result_1 = create_train_model(x_PCA.shape[1], x_train, y_train, x_test, y_test, epochs)

# SVD reduction

org_dim = x.shape[1]
tsvd = TruncatedSVD(org_dim - 1)
tsvd.fit(x)

# Save cumulative explained variance
cum_var = (np.cumsum(tsvd.explained_variance_ratio_))
n_comp = [i for i in range(1, org_dim)]
dim_sgd = len([i for i in cum_var if i < Threshold])
print(dim_sgd)

tsvd = TruncatedSVD(n_components=dim_sgd)
x_tsvd = tsvd.fit_transform(x)


x_train, x_test, y_train, y_test = train_test_split(x_tsvd, y, train_size=0.8, random_state=21)


history_2, result_2 = create_train_model(x_tsvd.shape[1], x_train, y_train, x_test, y_test, epochs)



# Final Graphs


plt.plot(history.history['accuracy'])
plt.plot(history_1.history['accuracy'])
plt.plot(history_2.history['accuracy'])
plt.title('Model Training accuracy file:' + file )
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend([str('Original ' + str(orig_feat) + ' feat.') , str('PCA 90% ' + str(dim_PCA) + ' feat.'), str('SVD 80% ' + str(dim_sgd) + ' feat.' )], loc='lower right')
plt.show()


plt.plot(history.history['val_accuracy'])
plt.plot(history_1.history['val_accuracy'])
plt.plot(history_2.history['val_accuracy'])
plt.title('Model validation accuracy file:' + file)
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend([str('Original ' + str(orig_feat) + ' feat.') , str('PCA 80% ' + str(dim_PCA) + ' feat.'), str('SVD 80% ' + str(dim_sgd) + ' feat.' )], loc='lower right')
plt.show()


