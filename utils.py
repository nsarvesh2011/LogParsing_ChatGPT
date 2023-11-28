from datasets import load_dataset
import pandas as pd

# Function to load training data from a specified directory, dataset, and shot
def load_train_data(r_dir=".", dataset="Apache", shot=4):
    dataset = load_dataset('json', data_files=f'{r_dir}/{dataset}/{4}shot/1.json')  # Loading the dataset from a JSON file
    examples = [(x['text'], x['label']) for x in dataset['train']]     # Extracting text and labels from the training examples
    return examples

# Function to load test data from a specified directory and dataset
def load_test_data(r_dir=".", dataset="Apache"):
    logs = pd.read_csv(f"{r_dir}/{dataset}/{dataset}_2k.log_structured_corrected.csv")  # Reading a CSV file containing log data using pandas
    return logs.Content.tolist()      # Extracting log content and converting it to a list

# Function to get log messages for training and testing
def get_log_messages(r_dir, dataset, shot=0):
    train, test = [], []     # Initializing empty lists for training and test data
    if shot > 0:             # Checking if shot is greater than 0
        demos = load_train_data(f"{r_dir}/dataset", dataset, shot)      # Loading training data and extracting text and labels
        for demo in demos:
            train.append((demo[0].strip(), demo[1].strip()))
    test_logs = load_test_data(f"{r_dir}/dataset", dataset)              # Loading test data and extracting log content
    for i, log in enumerate(test_logs):
        test.append(log.strip())

    return train, test                                                   # Returning the training and test data