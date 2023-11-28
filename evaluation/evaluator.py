import pandas as pd
import scipy.special
from nltk.metrics.distance import edit_distance
from sklearn.metrics import accuracy_score
import numpy as np

# Define a function to evaluate the performance of a log parsing system
def evaluate(groundtruth, parsedresult):
    # Read the groundtruth and parsed log files into pandas dataframes
    df_groundtruth = pd.read_csv(groundtruth)
    df_parsedlog = pd.read_csv(parsedresult, index_col=False)
    # df_groundtruth['EventTemplate'] = df_groundtruth['EventTemplate'].str.lower()

    # Read the groundtruth and parsed log files into pandas dataframes
    null_logids = df_groundtruth[~df_groundtruth['EventTemplate'].isnull()].index
    df_groundtruth = df_groundtruth.loc[null_logids]
    df_parsedlog = df_parsedlog.loc[null_logids]
    # Calculate accuracy using exact string matching
    accuracy_exact_string_matching = accuracy_score(np.array(df_groundtruth.EventTemplate.values, dtype='str'),
                                                    np.array(df_parsedlog.EventTemplate.values, dtype='str'))

    # Calculate edit distance between each pair of groundtruth and parsed log event templates
    edit_distance_result = []
    for i, j in zip(np.array(df_groundtruth.EventTemplate.values, dtype='str'),
                    np.array(df_parsedlog.EventTemplate.values, dtype='str')):
        edit_distance_result.append(edit_distance(i, j))

    edit_distance_result_mean = np.mean(edit_distance_result)   # Calculate mean edit distance
    edit_distance_result_std = np.std(edit_distance_result)     # Calculate standard deviation of edit distance

    # Calculate precision, recall, F1 measure, and group accuracy
    (precision, recall, f_measure, accuracy_PA) = get_accuracy(df_groundtruth['EventTemplate'],
                                                               df_parsedlog['EventTemplate'])

    # Identify unseen events in the groundtruth
    unseen_events = df_groundtruth.EventTemplate.value_counts()
    df_unseen_groundtruth = df_groundtruth[df_groundtruth.EventTemplate.isin(unseen_events.index[unseen_events.eq(1)])]
    df_unseen_parsedlog = df_parsedlog[df_parsedlog.LineId.isin(df_unseen_groundtruth.LineId.tolist())]
    n_unseen_logs = len(df_unseen_groundtruth)
    
    # Calculate accuracy for unseen events
    if n_unseen_logs == 0:
        unseen_PA = 0
    else:
        unseen_PA = accuracy_score(np.array(df_unseen_groundtruth.EventTemplate.values, dtype='str'),
                                   np.array(df_unseen_parsedlog.EventTemplate.values, dtype='str'))
        
    # Print evaluation metrics
    print(
        'Precision: %.4f, Recall: %.4f, F1_measure: %.4f, Group Accuracy: %.4f, Message-Level Accuracy: %.4f, Edit Distance: %.4f' % (
            precision, recall, f_measure, accuracy_PA, accuracy_exact_string_matching, edit_distance_result_mean))

    # Return evaluation metrics
    return accuracy_PA, accuracy_exact_string_matching, edit_distance_result_mean, edit_distance_result_std, unseen_PA, n_unseen_logs

# Define a function to calculate precision, recall, F1 measure, and accuracy
def get_accuracy(series_groundtruth, series_parsedlog, debug=False):
    
    # Calculate the number of true pairs in the groundtruth
    series_groundtruth_valuecounts = series_groundtruth.value_counts()
    real_pairs = 0
    for count in series_groundtruth_valuecounts:
        if count > 1:
            real_pairs += scipy.special.comb(count, 2)

    # Calculate the number of pairs in the parsed log
    series_parsedlog_valuecounts = series_parsedlog.value_counts()
    parsed_pairs = 0
    for count in series_parsedlog_valuecounts:
        if count > 1:
            parsed_pairs += scipy.special.comb(count, 2)

    # Initialize counters for accurate pairs and accurate events
    accurate_pairs = 0
    accurate_events = 0  # determine how many lines are correctly parsed
    for parsed_eventId in series_parsedlog_valuecounts.index:    # Iterate over unique event IDs in the parsed log
        logIds = series_parsedlog[series_parsedlog == parsed_eventId].index
        series_groundtruth_logId_valuecounts = series_groundtruth[logIds].value_counts()
        error_eventIds = (parsed_eventId, series_groundtruth_logId_valuecounts.index.tolist())
        error = True
        if series_groundtruth_logId_valuecounts.size == 1:       # Check if there is only one groundtruth event ID for the log IDs
            groundtruth_eventId = series_groundtruth_logId_valuecounts.index[0]
            if logIds.size == series_groundtruth[series_groundtruth == groundtruth_eventId].size:
                accurate_events += logIds.size
                error = False
        # If there is an error and debugging is enabled, print error information
        if error and debug:
            print('(parsed_eventId, groundtruth_eventId) =', error_eventIds, 'failed', logIds.size, 'messages')
        
        # Count accurate pairs
        for count in series_groundtruth_logId_valuecounts:
            if count > 1:
                accurate_pairs += scipy.special.comb(count, 2)

    # Calculate precision, recall, F1 measure, and accuracy
    precision = float(accurate_pairs) / parsed_pairs
    recall = float(accurate_pairs) / real_pairs
    f_measure = 2 * precision * recall / (precision + recall)
    accuracy = float(accurate_events) / series_groundtruth.size
    return precision, recall, f_measure, accuracy