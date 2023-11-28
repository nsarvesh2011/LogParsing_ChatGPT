from evaluation.evaluator import evaluate

out_dir = "outputs/ChatGPT/4_shot"

datasets = ['Android']
#compares our output results with the ground truth
if __name__ == '__main__':
    for dataset in datasets:
        evaluate(f"dataset/{dataset}/{dataset}_2k.log_structured_corrected.csv",
                 f"{out_dir}/{dataset}_2k.log_structured_adjusted.csv")
