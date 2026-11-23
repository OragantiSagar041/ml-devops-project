
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

from preprocess import preprocess_data


def train_model():

    # Get preprocessed data
    X_train, X_test, y_train, y_test = preprocess_data()

    # Create the model
    model = LogisticRegression(max_iter=200)

    # Train the model
    model.fit(X_train, y_train)

    # Predict test data
    predictions = model.predict(X_test)

    # Calculate accuracy
    accuracy = accuracy_score(y_test, predictions)

    print("Model Training Completed")
    print("Model Accuracy:", accuracy)


if __name__ == "__main__":
    train_model()
