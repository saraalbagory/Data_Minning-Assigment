import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import MinMaxScaler
from collections import Counter
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

def read_file(file_path,percentage):
    data=pd.read_csv(file_path)
    percentage=percentage/100
    sampled_data = data.sample(frac=percentage)
    sampled_data = sampled_data.reset_index(drop=True)

    # Drop rows with any missing values
    df_clean = sampled_data.dropna()
    
    target_column = df_clean.columns[-1]

    # Separate features and target
    features= df_clean.drop(columns=[target_column])
    classfications= df_clean[target_column]
    

# Convert all  non-numeric columns to numeric using label encoding
    features_encoded = features.copy()
    for column in features_encoded.columns:
        if features_encoded[column].dtype == 'object':
            le = LabelEncoder()
            features_encoded[column] = le.fit_transform(features_encoded[column])

    # Also encode the target (ckd/notckd)
    le_target = LabelEncoder()
    labels_encoded = le_target.fit_transform(classfications)
    # Normalize the features
    scaler = MinMaxScaler()
    normalized_features= scaler.fit_transform(features_encoded)
    features_encoded = pd.DataFrame(normalized_features, columns=features_encoded.columns)
    print("Encoded labels:", labels_encoded)
    print("Encoded features:", features_encoded)
    return normalized_features, labels_encoded


def euclidean_distance(point1, point2):
    return np.sqrt(np.sum((point1 - point2) ** 2))


class KNN:
    def __init__(self, k=3):
        self.k = k
    
    def fit(self, X, y):
        self.X_train = X
        self.y_train = y

    def predict(self, X):
        predictions = []
        for point in X:
            prediction = self._predict(point)
            predictions.append(prediction)
        return np.array(predictions)
        
    def _predict(self, point):
        distances=[euclidean_distance(point, train_point) for train_point in self.X_train]
        #This returns the indices that would sort the distances ,[:self.k] returns the first k indices
        k_indices=np.argsort(distances)[:self.k]
        #Get the labels of the k nearest neighbors
        k_nearest_labels=[self.y_train[i] for i in k_indices]
        #Return the most common class label among the k neighbors "This counts how many times each label appears"
        most_common=Counter(k_nearest_labels).most_common()

        return most_common[0][0]


import numpy as np

class ANN(object):
    def __init__(self):
        self.input_size = 25
        self.output_size = 1
        self.hidden_size = 15

        # Weights
        self.w1 = np.random.randn(self.input_size, self.hidden_size) #wieghts between input and hidden layer (25*10)
        self.w2 = np.random.randn(self.hidden_size, self.output_size) # wieghts between hidden and output layer (10*1)

        # Biases
        self.b1 = np.zeros((1, self.hidden_size))
        self.b2 = np.zeros((1, self.output_size))

    def sigmoid(self,s, deriv=False):
        if(deriv==True):
            return s * (1-s)
        return 1/(1+np.exp(-s))

    
    def feed_forward(self, X):
        self.y1 = np.dot(X, self.w1) + self.b1 # w1*X + b1
        self.z1 = self.sigmoid(self.y1) # 1/(1+np.exp(-y1)) "Activation function"
        self.y2 = np.dot(self.z1, self.w2) + self.b2 # w2*z1 + b2
        output = self.sigmoid(self.y2) # 1/(1+np.exp(-y2)) "Activation function" "output layer"
        return output
    
    # Backpropagation
    def backprop(self, X, y, output, learning_rate=0.01):
        self.output_error = y - output
        self.output_delta = self.output_error * self.sigmoid(output, deriv=True)

        self.hidden_error = np.dot(self.output_delta, self.w2.T)
        self.z1_delta = self.hidden_error * self.sigmoid(self.z1, deriv=True)

        # Update weights and biases
        self.w2 += learning_rate * np.dot(self.z1.T, self.output_delta)
        self.b2 += learning_rate * np.sum(self.output_delta, axis=0, keepdims=True)

        self.w1 += learning_rate * np.dot(X.T, self.z1_delta)
        self.b1 += learning_rate * np.sum(self.z1_delta, axis=0, keepdims=True)
        # print("output_delta shape:", self.output_delta.shape)
        # print("w2.T shape:", self.w2.T.shape)




    def fit(self, X, y, epochs=1000, learning_rate=0.01):
        y = y.reshape(-1, 1)
        for epoch in range(epochs):
            output = self.feed_forward(X)
            self.backprop(X, y, output, learning_rate)
            if epoch % 100 == 0:
                loss = np.mean(np.square(y - output))
                print(f"Epoch {epoch}, Loss: {loss}")


    def predict(self, X):
        output = self.feed_forward(X)
        return np.round(output)
    

    def accuracy(self, X, y, predictions):
        # predictions = np.round( self.predict(X))
        if len(predictions) == 0:
            return 0
        # Ensure shapes match
        y = y.reshape(-1, 1)
        correct_predictions = np.sum(predictions.astype(int) == y)
        # Handle division by zero
        if len(y) == 0:
            return 0
        accuracy = correct_predictions / len(y)
        return accuracy

    

def main():
    # Read the file and get the normalized features and labels
    normalized_features, labels_encoded = read_file("data/Kidney_Disease Data for classification.csv", 100)
    
    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(normalized_features, labels_encoded, test_size=0.25, random_state=42)
    # print("X_train shape:", X_train)
    # print("y_train shape:", y_train)
    # print("X_test shape:", X_test)
    # print("y_test shape:", y_test)
    
    # Create a KNN classifier
    # knn = KNN(k=8)
    
    # # Fit the model
    # knn.fit(X_train, y_train)
    
    # # Make predictions on the test set
    # predictions = knn.predict(X_test)
    
    # # Calculate accuracy
    # accuracy = np.sum(predictions == y_test)/len(y_test)
    # print(f"Accuracy: {accuracy * 100:.2f}%")
    # Create an ANN classifier
    ann = ANN()
    ann.fit(X_train, y_train, epochs=150, learning_rate=0.0015)
    y_pred = ann.predict(X_test)
    print(classification_report(y_test, y_pred))
    accuracy = ann.accuracy(X_test, y_test, y_pred)
    print(f"Accuracy: {accuracy*100}%")
    
    # print(np.bincount(labels_encoded.flatten()))





main()