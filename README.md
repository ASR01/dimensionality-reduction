# Dimensionality Reduction

Just an exercise to be able to reduce the dimensionality of a ML Problem with tho datasets.

We will use a Sonar dataset or the Breast Cancer dataset that is available from kaggle. I just used this two to get binary classification and a set with no categorical values and do not have to perform OHE activities.

- Sonar case we can see that we are talking about a 60 features file.
- Breast cancer Dataset, we can see that we are talking about 29 features.

The supplied .py alloow to set the variance parameter to fine tune the dimensionality reduction.

Both methods PCA and SGM are capable of reduce the feature number by a given number. 

The file gets one of both datasets (it is configurable at the beginning) generates a DNN with a full layer of 512 units, a Dropout of 0.2 and a Sigmoid to make the final classification.

What the file does:

  1. Get the data from the file and divide them into x(features) and y(labels).
  2. Train a model with a DNN as explained above.
  3. After that performs a PCA analisys, keeping the features responsible of 80% of the variance.
    In this case for Sonar we stay with 10 variables and in the case of Breast-cancer whe remain with 3.
  4. We generate a new model with only the 10 and 3 features.
  5. We repeathe the steps 3 and 4 for the SGM method.


The results are the following:

for Breast Cancer dataset

![Image](./pics/BC_Acc.png)

and for the sonar dataset

![Image](./pics/Sonar_Acc.png)

We can see tht a massive reduction of 90% in the case of Breast Cancer and 83% in the case of the Sonar file in the computations needed does not change the result drastically. Even in the case of sonar it performs better. the the full-featured data.
