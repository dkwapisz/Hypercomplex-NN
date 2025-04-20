import pandas as pd

from analysis.MultiRunAnalysis import evaluation_by, read_phase_data, RUNS_TO_TEST, RUN_LABELS_SPLIT_5_PLUS

phases_to_compare = []

for phase in [f"phase{i}" for i in range(21, 21)]:
    phases_to_compare.append(read_phase_data(phase, RUNS_TO_TEST, only_hypercomplex=False))

def get_average_evaluation_between_phases_by_model(phases_data, phase_nums, run_labels, runs, evaluation_key="accuracy"):
    grouped_evaluations = [evaluation_by(data, runs, evaluation_key) for data in phases_data]
    average_evaluations = [{training_data_split: {group_by_value: sum(values) / len(values) for group_by_value, values in group_by_values.items()} for
                            training_data_split, group_by_values in grouped_evaluation.items()} for grouped_evaluation in grouped_evaluations]

    best_models = {}
    for phase_index, average_evaluation in enumerate(average_evaluations):
        for training_data_split, group_by_values in average_evaluation.items():
            for group_by_value, avg_evaluation in group_by_values.items():
                key = (training_data_split, group_by_value)
                if key not in best_models or avg_evaluation > best_models[key][f"average_{evaluation_key}"]:
                    best_models[key] = {"training_data_split": run_labels[runs.index(training_data_split)], "model": group_by_value,
                                        f"average_{evaluation_key}": avg_evaluation,
                                        "phase": f"phase{phase_nums[phase_index]}"}
    plot_data = list(best_models.values())

    df = pd.DataFrame(plot_data)

    df.sort_values(by=["model", "training_data_split", f"average_{evaluation_key}"], ascending=True, inplace=True)

    markdown_table = df.to_markdown(index=False)
    with open(f"between_phases_results/phase{'_'.join(map(str, phase_nums))}_by_average_{evaluation_key}.md", "w") as f:
        f.write(markdown_table)

    model_averages = df.groupby("model")[f"average_{evaluation_key}"].mean().reset_index().sort_values(by=[f"average_{evaluation_key}"], ascending=False, inplace=False)
    model_averages.columns = ["model", f"mean_average_{evaluation_key}"]

    model_averages_markdown = model_averages.to_markdown(index=False)
    with open(f"between_phases_results/mean_average_{evaluation_key}_per_model.md", "w") as f:
        f.write(model_averages_markdown)


get_average_evaluation_between_phases_by_model(phases_to_compare, (21), [70], RUNS_TO_TEST, "accuracy")
# get_average_evaluation_between_phases_by_model(phases_to_compare, (21), [70], RUNS_TO_TEST, "F1-Score")