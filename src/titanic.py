# Package imports
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
import sklearn.preprocessing as preprocessing

np.random.seed(1) # set a seed so that the results are consistent


def set_missing_ages(df):

    age_df = df[['Age','Fare', 'Parch', 'SibSp', 'Pclass']]

    known_age = age_df[age_df.Age.notnull()].as_matrix()
    unknown_age = age_df[age_df.Age.isnull()].as_matrix()

    y = known_age[:, 0]

    X = known_age[:, 1:]

    rfr = RandomForestRegressor(random_state=0, n_estimators=2000, n_jobs=-1)
    rfr.fit(X, y)

    predictedAges = rfr.predict(unknown_age[:, 1::])

    df.loc[ (df.Age.isnull()), 'Age' ] = predictedAges 

    return df, rfr

def set_Cabin_type(df):
    df.loc[ (df.Cabin.notnull()), 'Cabin' ] = 1
    df.loc[ (df.Cabin.isnull()), 'Cabin' ] = 0
    return df

def set_Sex(df):
    df.loc[ (df.Sex == 'male'), 'Sex' ] = 1
    df.loc[ (df.Sex == 'female'), 'Sex' ] = 0
    return df

def set_Embarked(df):
    df.loc[ (df.Embarked == 'S'), 'Embarked' ] = 2
    df.loc[ (df.Embarked == 'C'), 'Embarked' ] = 1
    df.loc[ (df.Embarked == 'Q'), 'Embarked' ] = 0
    return df

X = pd.read_csv("../data/train.csv")
X.info()
X, rfr = set_missing_ages(X)
X = set_Cabin_type(X)
X = set_Sex(X)

scaler = preprocessing.StandardScaler()
age_scale_param = scaler.fit(X['Age'])
X['Age'] = scaler.fit_transform(X['Age'], age_scale_param)
fare_scale_param = scaler.fit(X['Fare'])
X['Fare'] = scaler.fit_transform(X['Fare'], fare_scale_param)

Y = X['Survived']

X.drop(['PassengerId', 'Name', 'Ticket', 'Survived', 'Embarked'], axis = 1, inplace = True)
X = X.astype(float)
X = X.T
Y = Y.T

X = np.array(X)
Y = np.array(Y)

# GRADED FUNCTION: layer_sizes
def layer_sizes(X, Y):
    n_x = len(X) # size of input layer
    n_h = 4
    n_y = 1 # size of output layer
    return (n_x, n_h, n_y)

(n_x, n_h, n_y) = layer_sizes(X, Y)

print("The size of the input layer is: n_x = " + str(n_x))
print("The size of the hidden layer is: n_h = " + str(n_h))
print("The size of the output layer is: n_y = " + str(n_y))

def initialize_parameters(n_x, n_h, n_y):
    np.random.seed(2) # we set up a seed so that your output matches ours although the initialization is random.
    
    W1 = np.random.randn(n_h, n_x) * 0.01
    b1 = np.zeros((n_h, 1))
    W2 = np.random.randn(n_y, n_h) * 0.01
    b2 = np.zeros((n_y, 1))
    
    assert (W1.shape == (n_h, n_x))
    assert (b1.shape == (n_h, 1))
    assert (W2.shape == (n_y, n_h))
    assert (b2.shape == (n_y, 1))
    
    parameters = {"W1": W1,
                  "b1": b1,
                  "W2": W2,
                  "b2": b2}
    
    return parameters

parameters = initialize_parameters(n_x, n_h, n_y)

print("W1 = " + str(parameters["W1"]))
print("b1 = " + str(parameters["b1"]))
print("W2 = " + str(parameters["W2"]))
print("b2 = " + str(parameters["b2"]))

def sigmoid(X):
    Y = 1 / (1 + np.exp(-X))
    return Y

# GRADED FUNCTION: forward_propagation

def forward_propagation(X, parameters):
    W1 = parameters["W1"]
    b1 = parameters["b1"]
    W2 = parameters["W2"]
    b2 = parameters["b2"]

    Z1 = np.dot(W1, X) + b1
    A1 = np.tanh(Z1)
    Z2 = np.dot(W2, A1) + b2
    A2 = sigmoid(Z2)
    
    assert(A2.shape == (1, X.shape[1]))
    
    cache = {"Z1": Z1,
             "A1": A1,
             "Z2": Z2,
             "A2": A2}
    
    return A2, cache

# GRADED FUNCTION: compute_cost

def compute_cost(A2, Y, parameters):
    
    logprobs = np.multiply(np.log(A2), Y)
    cost = -np.sum(logprobs)
    
    cost = np.squeeze(cost)
    assert(isinstance(cost, float))
    
    return cost

# GRADED FUNCTION: backward_propagation

def backward_propagation(parameters, cache, X, Y):
    m = X.shape[1]
    
    W2 = parameters["W2"]

    A1 = cache["A1"]
    A2 = cache["A2"]

    dZ2 = A2 - Y
    dW2 = np.dot(dZ2, A1.T) / m
    db2 = np.sum(dZ2, axis = 1, keepdims = True) / m
    dZ1 = np.dot(W2.T, dZ2) * (1 - np.power(A1, 2))
    dW1 = np.dot(dZ1, X.T) / m
    db1 = np.sum(dZ1, axis = 1, keepdims = True) / m
    
    grads = {"dW1": dW1,
             "db1": db1,
             "dW2": dW2,
             "db2": db2}
    
    return grads

# GRADED FUNCTION: update_parameters

def update_parameters(parameters, grads, learning_rate = 0.6):
    W1 = parameters["W1"]
    b1 = parameters["b1"]
    W2 = parameters["W2"]
    b2 = parameters["b2"]
    
    dW1 = grads["dW1"]
    db1 = grads["db1"]
    dW2 = grads["dW2"]
    db2 = grads["db2"]

    W1 = W1 - learning_rate * dW1
    b1 = b1 - learning_rate * db1
    W2 = W2 - learning_rate * dW2
    b2 = b2 - learning_rate * db2
    
    parameters = {"W1": W1,
                  "b1": b1,
                  "W2": W2,
                  "b2": b2}
    
    return parameters

# GRADED FUNCTION: nn_model

def nn_model(X, Y, n_h, num_iterations = 10000, print_cost=False):
    np.random.seed(3)
    n_x = layer_sizes(X, Y)[0]
    n_y = layer_sizes(X, Y)[2]
    
    parameters = initialize_parameters(n_x, n_h, n_y)
    
    for i in range(0, num_iterations):
        A2, cache = forward_propagation(X, parameters)
        
        cost = compute_cost(A2, Y, parameters)
 
        grads = backward_propagation(parameters, cache, X, Y)
 
        parameters = update_parameters(parameters, grads)

        if print_cost and i % 1000 == 0:
            print ("Cost after iteration %i: %f" %(i, cost))

    return parameters

parameters = nn_model(X, Y, 4, num_iterations=10000, print_cost=True)
print("W1 = " + str(parameters["W1"]))
print("b1 = " + str(parameters["b1"]))
print("W2 = " + str(parameters["W2"]))
print("b2 = " + str(parameters["b2"]))

# GRADED FUNCTION: predict

def predict(parameters, X):
    A2, cache = forward_propagation(X, parameters)
    predictions = A2 > 0.5
    
    return predictions

X = pd.read_csv("../data/test.csv")
X.info()
X.loc[ (X.Fare.isnull()), 'Fare' ] = 0
tmp_df = X[['Age','Fare', 'Parch', 'SibSp', 'Pclass']]
null_age = tmp_df[X.Age.isnull()].as_matrix()
age = null_age[:, 1:]
predictedAges = rfr.predict(age)
X.loc[ (X.Age.isnull()), 'Age' ] = predictedAges
X = set_Cabin_type(X)
X = set_Sex(X)

scaler = preprocessing.StandardScaler()
age_scale_param = scaler.fit(X['Age'])
X['Age'] = scaler.fit_transform(X['Age'], age_scale_param)
fare_scale_param = scaler.fit(X['Fare'])
X['Fare'] = scaler.fit_transform(X['Fare'], fare_scale_param)

tmp = X.filter(regex='Pclass|Sex|Age|SibSp|Parch|Fare|Cabin')
tmp = tmp.astype(float)
tmp = tmp.T
tmp = np.array(tmp)

predictions = predict(parameters, tmp)
print(str(predictions))
print("predictions mean = " + str(np.mean(predictions)))
predictions = np.array(predictions)
predictions = predictions.astype(np.int32)
# X = np.array(X)
result = pd.DataFrame({'PassengerId':X['PassengerId'].as_matrix()})
prediction = pd.DataFrame(predictions.T)
prediction.columns = ['Survived']
result = pd.concat([result, prediction], axis = 1)
print(result)
result.to_csv("../data/logistic_regression_predictions.csv", index=False)
# prediction.to_csv("../data/logistic_regression_predictions.csv", index=False, columns = 1)