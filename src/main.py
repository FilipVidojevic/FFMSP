from src.ffmsp_optimizer import FFMSP_Optimizer
import json
import numpy as np
import time
import os
test_files = [
    # "data/datasets/100-300-001.txt",
    #"data/datasets/100-600-001.txt",
    #"data/datasets/100-800-001.txt",
    #"data/datasets/200-300-001.txt",
    #"data/datasets/200-600-001.txt",
    "data/datasets/200-800-001.txt"
]

def main():
    # problem = FFMSP_Optimizer("data/datasets/100-300-001.txt", t_scale=0.8)
    # best = problem.hybrid_ACO(90, 1000000)
    # print("Found solution: " + str(best))
    # print("OBJ: " + str(problem.objective_function(best)))
    # print("Done!")
    run_experiment()

def append_result(instance, avg_value):
        with open("results.csv", 'a') as f:
            f.write(f"{instance}, {avg_value}\n")

def run_experiment(config_path="configs/optimizer_strategies.json"):
    pr_strategies = ["random", "greedy"]
    ls_strategies = ["best_improvement", "first_improvement"]
    ls_after_pr_strategies = ["with_ls_after_pr", "without_ls_after_pr"]
    pr_strategies = ["left_to_right"]
    results = {}

    for strategy in pr_strategies:
        # Load the existing configuration
        # try:
        #     with open(config_path, 'r') as config_file:
        #         config = json.load(config_file)
        # except (FileNotFoundError, json.JSONDecodeError):
        #     config = {}  # Start with an empty config if the file doesn't exist or is invalid

        # # Update the strategy
        # config["local_search_strategy"] = strategy

        # # Save the updated configuration
        # with open(config_path, 'w') as config_file:
        #     json.dump(config, config_file, indent=4)

        # Run the optimizer multiple times
        for test_file in test_files:
            testrun_name = f"{test_file}_{strategy}"
            results[testrun_name] = []
            for i in range(10):  # Run each configuration 10 times
                optimizer = FFMSP_Optimizer(file_path=test_file, config_path=config_path, t_scale=0.8)
                best_solution = optimizer.hybrid_ACO(tlim_aco=90, runlim_aco=11111)
                objective_value = optimizer.objective_function(best_solution)

                results[testrun_name].append(objective_value)
                print(f"Run {test_file} for strategy '{strategy}' completed with objective value: {objective_value}")

    # Calculate and print averages
    for key, value in results.items():
        avg = np.mean(value)
        print(f"Average objective value for '{key}': {avg}")
        append_result(key, avg)

    return results

if __name__=="__main__":
    main()