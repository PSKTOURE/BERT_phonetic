import torch
from collections import defaultdict
import evaluate
import numpy as np
from src.utils import load_glue_dataset_from_dir, task_to_metric
from concurrent.futures import ProcessPoolExecutor
import time

tasks = [
    "rte",
    "sst2",
    "qqp",
    "mnli",
    "qnli",
    "cola",
    "mrpc",
    "stsb",
    "wnli",
]
# task_to_metric = {
#     "cola": ["matthews_correlation"],
#     "mnli_matched": ["accuracy"],
#     "mnli_mismatched": ["accuracy"],
#     "mrpc": ["accuracy", "f1"],
#     "qnli": ["accuracy"],
#     "qqp": ["accuracy", "f1"],
#     "rte": ["accuracy"],
#     "sst2": ["accuracy"],
#     "stsb": ["pearson", "spearmanr"],
#     "wnli": ["accuracy"],
# }


def run_iteration():
    iteration_results = defaultdict(list)

        # Load the dataset
    for task in tasks:
        if task == "mnli":
            dataset = load_glue_dataset_from_dir("mnli")
            matched_dataset = dataset["validation_matched"]
            mismatched_dataset = dataset["validation_mismatched"]
            matched_labels = matched_dataset["label"]
            mismatched_labels = mismatched_dataset["label"]
            matched_random_predictions = torch.randint(0, 3, (len(matched_dataset),))
            mismatched_random_predictions = torch.randint(
                0, 3, (len(mismatched_dataset),)
            )
            metric = evaluate.load("glue", task)
            matched_result = metric.compute(
                predictions=matched_random_predictions, references=matched_labels
            )
            mismatched_result = metric.compute(
                predictions=mismatched_random_predictions, references=mismatched_labels
            )
            iteration_results["mnli_matched"].append(list(matched_result.values()))
            iteration_results["mnli_mismatched"].append(
                list(mismatched_result.values())
            )
            continue

        dataset = load_glue_dataset_from_dir(task)
        dataset = dataset["validation"]
        if task != "stsb":
            # Classification task
            labels = dataset.features["label"].names
            random_predictions = torch.randint(0, len(labels), (len(dataset),))
        else:
            # Regression task (stsb)
            random_predictions = torch.rand(len(dataset)) * 5

        metric = evaluate.load(task_to_metric[task][0])
        result = metric.compute(
            predictions=random_predictions, references=dataset["label"]
        )
        iteration_results[task].append(list(result.values()))

    return iteration_results


def run_parallel_iterations():
    all_results = defaultdict(list)

    # Use ProcessPoolExecutor for parallelism
    with ProcessPoolExecutor() as executor:
        futures = [executor.submit(run_iteration) for _ in range(10)]
        for future in futures:
            iteration_result = future.result()
            for task in tasks:
                if task == "mnli":
                    all_results["mnli_matched"].append(iteration_result["mnli_matched"][0])
                    all_results["mnli_mismatched"].append(iteration_result["mnli_mismatched"][0])
                    continue
                all_results[task].append(iteration_result[task][0])

    # Compute the mean of results over iterations
    for key, value in all_results.items():
        all_results[key] = np.mean(value, axis=0)
    
    final_results = defaultdict(dict)
    for task in tasks:
        if task == "mnli":
            final_results["mnli_matched"] = dict(zip(task_to_metric[task], all_results["mnli_matched"]))
            final_results["mnli_mismatched"] = dict(zip(task_to_metric[task], all_results["mnli_mismatched"]))
            continue
        final_results[task] = dict(zip(task_to_metric[task], all_results[task]))
        
    # convert to float
    for task in final_results.keys():
        final_results[task] = {metric: float(value) for metric, value in final_results[task].items()}

    return final_results


if __name__ == "__main__":
    start = time.time()
    results = run_parallel_iterations()
    with open("logs/random_classifier.tsv", "w") as f:
        for task, metrics in results.items():
            f.write(f"Random_Classifier\t{task}\t{metrics}\n")
    print(f"Time taken: {time.time() - start} seconds")

