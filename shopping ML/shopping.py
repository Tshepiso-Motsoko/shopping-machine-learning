import csv
import sys
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier

# This is the proportion of the dataset to include in the test split.
TEST_SIZE = 0.4

def main():
    # Check command-line arguments
    if len(sys.argv) != 2:
        sys.exit("Usage: python shopping.py data")

    # Load data from the CSV file provided in the command line argument
    evidence, labels = load_data(sys.argv[1])

    # Split the data into a training set and a testing set
    X_train, X_test, y_train, y_test = train_test_split(
        evidence, labels, test_size=TEST_SIZE
    )

    # Train the k-nearest neighbors model using the training data
    model = train_model(X_train, y_train)

    # Use the trained model to predict outcomes on the testing set
    predictions = model.predict(X_test)

    # Evaluate the model's performance and print sensitivity and specificity
    sensitivity, specificity = evaluate(y_test, predictions)

    # Print results to the console
    print(f"Correct: {(y_test == predictions).sum()}")
    print(f"Incorrect: {(y_test != predictions).sum()}")
    print(f"True Positive Rate: {100 * sensitivity:.2f}%")
    print(f"True Negative Rate: {100 * specificity:.2f}%")

def load_data(filename):
    """
    Load shopping data from a CSV file `filename` and return a tuple (evidence, labels).
    Convert data to the appropriate numeric types and encode categorical variables.
    """
    # Initialize lists for evidence and labels
    evidence = []
    labels = []

    # Open the CSV file for reading
    with open(filename) as f:
        reader = csv.reader(f)
        next(reader)  # Skip the header row

        # Iterate over the rows in the CSV file
        for row in reader:
            # Process and append the evidence and label for each row
            evidence.append(process_evidence(row[:-1]))
            labels.append(process_label(row[-1]))

    # Return the tuple containing lists of evidence and labels
    return (evidence, labels)

def process_evidence(row):
    """
    Convert a row of raw CSV data into a numerical evidence list.
    """
    month_index = ["Jan", "Feb", "Mar", "Apr", "May", "June", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"].index(row[10])
    visitor_type = 1 if row[15] == "Returning_Visitor" else 0
    weekend = 1 if row[16] == "TRUE" else 0

    # Convert each column to the appropriate type
    return [
        int(row[0]),  # Administrative
        float(row[1]),  # Administrative_Duration
        int(row[2]),  # Informational
        float(row[3]),  # Informational_Duration
        int(row[4]),  # ProductRelated
        float(row[5]),  # ProductRelated_Duration
        float(row[6]),  # BounceRates
        float(row[7]),  # ExitRates
        float(row[8]),  # PageValues
        float(row[9]),  # SpecialDay
        month_index,  # Month
        int(row[11]),  # OperatingSystems
        int(row[12]),  # Browser
        int(row[13]),  # Region
        int(row[14]),  # TrafficType
        visitor_type,  # VisitorType
        weekend  # Weekend
    ]

def process_label(label):
    """
    Convert the label from the CSV row into a numerical value.
    """
    return 1 if label == "TRUE" else 0

def train_model(evidence, labels):
    """
    Train a k-nearest neighbors model (k=1) on the evidence and labels.
    """
    # Initialize the classifier with k=1
    model = KNeighborsClassifier(n_neighbors=1)

    # Train the classifier with the evidence and labels
    model.fit(evidence, labels)

    # Return the trained model
    return model

def evaluate(labels, predictions):
    """
    Calculate the sensitivity and specificity of the model predictions.
    """
    true_positives = sum([1 if label == prediction == 1 else 0 for label, prediction in zip(labels, predictions)])
    true_negatives = sum([1 if label == prediction == 0 else 0 for label, prediction in zip(labels, predictions)])
    false_negatives = sum([1 if label == 1 and prediction == 0 else 0 for label, prediction in zip(labels, predictions)])
    false_positives = sum([1 if label == 0 and prediction == 1 else 0 for label, prediction in zip(labels, predictions)])

    # Calculate sensitivity and specificity
    sensitivity = true_positives / (true_positives + false_negatives)
    specificity = true_negatives / (true_negatives + false_positives)

    # Return the sensitivity and specificity as a tuple
    return (sensitivity, specificity)

if __name__ == "__main__":
    main()
