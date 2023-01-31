# dataScience
To calculate housing value using a machine learning algorithm, you would need to follow these steps:
    
Collect data: Gather information about the properties, including their location, size, number of rooms, age, and other relevant features.
libraries required 

```powershell
  pip install numpy
  pip install pandas
  pip install matplotlib
  pip install sklearn
```
    
Clean and preprocess the data: Handle missing values, normalize the features, and convert categorical variables into numerical ones.

``` python 
    #importing files
    df = pd.read_csv('./dataset/housing.csv')
```
    
Split the data into training and test sets: Divide the data into two sets, one for training the model and one for evaluating its performance.

``` python 
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
```
    
Train the model: Choose an appropriate machine learning algorithm (e.g. linear regression, decision tree, random forest, etc.) and train it on the training set.
I've used random forest algorithm to prodict train the model

Random forest brief explanatory : https://towardsdatascience.com/understanding-random-forest-58381e0602d2
    
Evaluate the model: Use the test set to evaluate the performance of the model and make adjustments as necessary.
    
Make predictions: Use the trained model to make predictions on new housing data and calculate the estimated housing value.
