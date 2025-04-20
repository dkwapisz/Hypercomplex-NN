import os
import json
import numpy as np
from collections import defaultdict
from scipy.stats import iqr
import matplotlib.pyplot as plt
import seaborn as sns

def calculate_accuracy_statistics(base_path):
    algebra_stats = defaultdict(list)

    run1_path = os.path.join(base_path, "run1", "results")
    for subfolder in os.listdir(run1_path):
        subfolder_path = os.path.join(run1_path, subfolder)
        training_file = os.path.join(subfolder_path, "training.json")

        if os.path.isfile(training_file):
            with open(training_file, "r") as f:
                data = json.load(f)
                algebra = data.get("model_name", {}).get("algebra")
                accuracy = data.get("evaluation_result", {}).get("accuracy")

                if algebra and accuracy is not None:
                    algebra_stats[algebra].append(accuracy)

    results = {}
    for algebra, accuracies in algebra_stats.items():
        accuracies = np.array(accuracies)
        results[algebra] = {
            "COUNT": len(accuracies),
            "MAX": np.max(accuracies),
            "MIN": np.min(accuracies),
            "AVG": np.mean(accuracies),
            "MEDIAN": np.median(accuracies),
            "STD": np.std(accuracies),
            "IQR": iqr(accuracies),
            "RANGE": np.max(accuracies) - np.min(accuracies)
        }

    return results, algebra_stats


def cohen_d(x, y):
    x, y = np.array(x), np.array(y)
    nx, ny = len(x), len(y)
    pooled_std = np.sqrt(((nx - 1) * np.std(x, ddof=1) ** 2 + (ny - 1) * np.std(y, ddof=1) ** 2) / (nx + ny - 2))
    return (np.mean(x) - np.mean(y)) / pooled_std

def generate_plots(algebra_stats, phase):
    plt.figure(figsize=(8, 6))
    data = []
    labels = []

    for algebra, accuracies in algebra_stats.items():
        data.append(np.array(accuracies))
        labels.append(algebra)

    sns.boxplot(data=data, orient="v")
    plt.title("Boxplot for transformation permutations")
    plt.ylabel("Accuracy")
    plt.xticks(ticks=range(len(labels)), labels=labels, rotation=45)
    plt.tight_layout()
    plt.savefig(f"{phase}/all_algebras_boxplot.png")
    plt.close()


def print_results(phase, plots):
    statistics, raw = calculate_accuracy_statistics(phase)

    if plots:
        generate_plots(raw, phase)

    print(f"Statistics for {phase}:")
    for algebra, stats in sorted(statistics.items(), key=lambda x: -x[1]["AVG"]):
        print(f"\nAlgebra: {algebra}")
        for stat_name, value in stats.items():
            print(f"  {stat_name}: {value:.6f}")

    if len(raw) == 2:
        keys = list(raw.keys())
        d = cohen_d(raw[keys[0]], raw[keys[1]])
        print(f"Cohen's d between {keys[0]} and {keys[1]}: {d:.4f}")
        if abs(d) < 0.2:
            interpretation = "negligible"
        elif abs(d) < 0.5:
            interpretation = "small"
        elif abs(d) < 0.8:
            interpretation = "medium"
        else:
            interpretation = "large"
        print(f"Effect size interpretation: {interpretation}")

    print("\n" + "-" * 40 + "\n")


print_results("phase21", plots=False)
print_results("phase22", plots=False)
