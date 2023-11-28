import os
from chat import ChatGPT, config
from utils import get_log_messages
import pandas as pd
from collections import Counter
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor
from dataset.data_loader import load_train_data

datasets = ['HDFS', 'Proxifier','Apache']

MSG_LEN = 1


def zero_shot_benchmark(model, prompt_template, dataset, out_dir="."):    
    chat = ChatGPT(model=model, prompt=prompt_template)   # Create a ChatGPT instance with the specified model and prompt template
    _, test = get_log_messages("./", dataset, 0)          # Load test data for the specified dataset
    log_chunks = []                                       # Load test data for the specified dataset
    for i in tqdm(range(len(test) // MSG_LEN)):
        log_chunks.append(test[i * MSG_LEN: (i + 1) * MSG_LEN])
    with ThreadPoolExecutor(max_workers=16) as executor:  # Use ThreadPoolExecutor to parallelize the generation of responses
        templates = list(
            tqdm(executor.map(lambda chunk: chat.get_response(chunk, request_type=MSG_LEN == 1), log_chunks),
                 total=len(log_chunks)))                  # Generate responses for each chunk of test data
        print("Completed!")
    os.makedirs("logs", exist_ok=True)                    # Create a directory for logs if it doesn't exist
    with open(f"logs/{dataset}_{out_dir}.log", mode="w") as f:
        [f.write(x[1] + "\n =================== \n") for x in templates]         # Write generated responses to a log file
    templates = [x[0] for x in templates]                 
    if MSG_LEN > 1:                                       # If MSG_LEN is greater than 1, flatten the list of templates
        templates = sum(templates, [])
    unique_templates = Counter(templates).items()         # Count the occurrences of each unique template
    logs_df = pd.read_csv(f"dataset/{dataset}/{dataset}_2k.log_structured_corrected.csv")  # Read the original log data from a CSV file
    logs_df.EventTemplate = pd.Series(templates)          # Replace the EventTemplate column with the generated templates
    temp_df = pd.DataFrame(unique_templates, columns=['EventTemplate', 'Occurrences'])    # Create a DataFrame with unique templates and their occurrences
    os.makedirs(f"outputs/{out_dir}", exist_ok=True)     # Create directories for outputs if they don't exist
    
    # Save the modified log data and the template occurrences to CSV files
    logs_df.to_csv(f"outputs/{out_dir}/{dataset}_2k.log_structured.csv")
    temp_df.to_csv(f"outputs/{out_dir}/{dataset}_2k.log_templates.csv")


def few_shot_benchmark(model, demo, prompt_template, demo_format, demo_inst, dataset, out_dir="."):
    # Create a ChatGPT instance with the specified model, prompt template, and demo information
    chat = ChatGPT(model=model, prompt=prompt_template, demo_format=demo_format, demo_instruct=demo_inst)
    _, test = get_log_messages("./", dataset, 0)   # Load test data for the specified dataset
    log_chunks = []
    for i in tqdm(range(len(test) // MSG_LEN)):
        log_chunks.append(test[i * MSG_LEN: (i + 1) * MSG_LEN])   # Divide the test data into chunks of size MSG_LEN
    with ThreadPoolExecutor(max_workers=8) as executor:
        templates = list(                                         # Generate responses for each chunk of test data using the provided demos
            tqdm(executor.map(lambda chunk: chat.get_response(chunk, demos=demo), log_chunks), total=len(log_chunks)))
        print("Completed!")                                       # Use ThreadPoolExecutor to parallelize the generation of response   
    os.makedirs("logs", exist_ok=True)
    with open(f"logs/{dataset}_{out_dir}.log", mode="w") as f:
        [f.write(x[1] + "\n =================== \n") for x in templates]
    templates = [x[0] for x in templates]
    unique_templates = Counter(templates).items()
    logs_df = pd.read_csv(f"dataset/{dataset}/{dataset}_2k.log_structured_corrected.csv")
    logs_df.EventTemplate = pd.Series(templates)
    temp_df = pd.DataFrame(unique_templates, columns=['EventTemplate', 'Occurrences'])
    os.makedirs(f"outputs/{out_dir}", exist_ok=True)
    logs_df.to_csv(f"outputs/{out_dir}/{dataset}_2k.log_structured.csv")
    temp_df.to_csv(f"outputs/{out_dir}/{dataset}_2k.log_templates.csv")


if __name__ == '__main__':
    """ zero-shot benchmark
    """
    prompt = config['ZERO_SHOT_ENHANCE_PROMPT']                # Define the zero-shot prompt
    print(prompt['prompt'], "-" * 5, prompt['desc'])
    for dname in datasets:                                    # Iterate over datasets and perform zero-shot benchmarking
        print(f"============== {dname} ==============")
        zero_shot_benchmark(config['MODEL'], prompt['prompt'], dname, f"{prompt['id']}")

    """ few-shot benchmark
    """
    # prompt = config['FEW_SHOT_PROMPT']
    # print(prompt['prompt'])
    # for shot in [4]:
    #     print(f"************ {shot} shot ************")
    #     for dname in datasets:
    #         print(f"============== {dname} ==============")
    #         demos = load_train_data(r_dir="./dataset", dataset=dname, shot=shot)
    #         few_shot_benchmark(config['MODEL'], demos, prompt['prompt'], prompt['demo_format'],
    #                            prompt['demo_instruct'], dname, f"{prompt['id']}_{shot}")
