import json, os
import plotly.express as plt_exp
import pandas as pd

from analysis.SingleRunAnalysis import get_formatted_model_name

# ------------------- Static parameters -------------------
JSON_RESULT_FILE = "training.json"
RUNS_TO_TEST = ["run1", "run2", "run3", "run4", "run5", "run6", "run7", "run8", "run9", "run10"]
PHASE_1 = "phase1"
PHASE_2 = "phase2"
PHASE_3 = "phase3"
PHASE_4 = "phase4"
# ---------------------------------------------------------

def read_phase_data(run_phase, runs, only_hypercomplex=False):
    data = {run: [] for run in runs}

    for run in runs:
        trained_models = os.listdir(os.path.join(run_phase, run, "results"))
        for model in trained_models:
            file_path = os.path.join(run_phase, run, "results", model, "training.json")
            if os.path.exists(file_path):
                with open(file_path, 'r') as f:
                    model_data = json.load(f)
                    if only_hypercomplex:
                        if model_data.get("model_name").get("type") == "HyperComplex":
                            data[run].append(model_data)
                    else:
                        data[run].append(model_data)

    return data


def group_evaluation_by(data, runs, group_by_key, evaluation_key):
    grouped_evaluation = {run: {} for run in runs}

    for run_name, run in data.items():
        for model in run:
            evaluation_value = model["evaluation_result"][evaluation_key]
            group_by_value = model["model_name"][group_by_key]

            if group_by_key == "algebra" and group_by_value == "":
                group_by_value = "CNN"

            if group_by_value not in grouped_evaluation[run_name]:
                grouped_evaluation[run_name][group_by_value] = []
            grouped_evaluation[run_name][group_by_value].append(evaluation_value)

    return grouped_evaluation

def get_average_evaluation_grouped_by(data, runs, run_phase, group_by_key, evaluation_key="accuracy"):
    grouped_evaluation = group_evaluation_by(data, runs, group_by_key, evaluation_key)

    average_evaluation = {run_name: {group_by_value: sum(values) / len(values) for group_by_value, values in group_by_values.items()} for
                          run_name, group_by_values in grouped_evaluation.items()}

    plot_data = []
    for run_name, group_by_values in average_evaluation.items():
        for group_by_value, avg_evaluation in group_by_values.items():
            plot_data.append({"run_name": run_name, f"{group_by_key}": group_by_value, f"average_{evaluation_key}": avg_evaluation})

    df = pd.DataFrame(plot_data)

    fig = plt_exp.line(df, x="run_name", y=f"average_{evaluation_key}", color=f"{group_by_key}",
                       title=f"Average {evaluation_key} per {group_by_key} - {run_phase}",
                       labels={"run_name": "Run Number", f"average_{evaluation_key}": f"Average {evaluation_key}",
                               f"{group_by_key}": f"{group_by_key}"})

    fig.update_layout(xaxis_tickangle=-45, height=600, width=1200)
    fig.write_image(os.path.join(run_phase, f"avg_{evaluation_key}_by_{group_by_key}.png"))

def get_average_evaluation_between_phases_grouped_by(data_phase1, data_phase2, phase_nums, runs, group_by_key, evaluation_key="accuracy"):
    grouped_evaluation_phase1 = group_evaluation_by(data_phase1, runs, group_by_key, evaluation_key)
    grouped_evaluation_phase2 = group_evaluation_by(data_phase2, runs, group_by_key, evaluation_key)

    average_evaluation_phase1 = {run_name: {group_by_value: sum(values) / len(values) for group_by_value, values in group_by_values.items()} for
                                 run_name, group_by_values in grouped_evaluation_phase1.items()}
    average_evaluation_phase2 = {run_name: {group_by_value: sum(values) / len(values) for group_by_value, values in group_by_values.items()} for
                                 run_name, group_by_values in grouped_evaluation_phase2.items()}

    plot_data = []
    for run_name, group_by_values in average_evaluation_phase1.items():
        for group_by_value, avg_evaluation in group_by_values.items():
            plot_data.append({"run_name": run_name, f"{group_by_key}": group_by_value, f"average_{evaluation_key}": avg_evaluation, "phase": f"phase{phase_nums[0]}"})

    for run_name, group_by_values in average_evaluation_phase2.items():
        for group_by_value, avg_evaluation in group_by_values.items():
            plot_data.append({"run_name": run_name, f"{group_by_key}": group_by_value, f"average_{evaluation_key}": avg_evaluation, "phase": f"phase{phase_nums[1]}"})

    df = pd.DataFrame(plot_data)

    fig = plt_exp.line(df, x="run_name", y=f"average_{evaluation_key}", color=f"{group_by_key}", line_dash="phase",
                       title=f"Average {evaluation_key} per {group_by_key} (Phase Comparison)",
                       labels={"run_name": "Run Number", f"average_{evaluation_key}": f"Average {evaluation_key}",
                               f"{group_by_key}": f"{group_by_key}", "phase": "Phase"})

    fig.update_layout(xaxis_tickangle=-45, height=600, width=1200)
    fig.write_image(os.path.join("between_phases_results", f"phase{phase_nums[0]}_phase{phase_nums[1]}_avg_{evaluation_key}_by_{group_by_key}_comparison.png"))


def plot_stacked_average_bar_for_phase(data, phase, evaluation_key):
    model_evaluation = {}

    for run_name, run_data in data.items():
        for model in run_data:
            model_name = get_formatted_model_name(model)
            evaluation_value = model["evaluation_result"][evaluation_key]
            if model_name not in model_evaluation:
                model_evaluation[model_name] = []
            model_evaluation[model_name].append(evaluation_value)

    model_evaluation_avg = {model: sum(values) / len(values) for model, values in model_evaluation.items()}

    plot_data = [{"Model": model, evaluation_key: avg_value} for model, avg_value in model_evaluation_avg.items()]

    df = pd.DataFrame(plot_data).sort_values(by=evaluation_key, ascending=True)

    fig = plt_exp.bar(df, x="Model", y=evaluation_key,
                      title=f"Average {evaluation_key} per Model",
                      labels={"Model": "Model", evaluation_key: evaluation_key},
                      text_auto=True)

    fig.update_layout(xaxis_tickangle=-45, height=900, width=1800)
    fig.write_image(os.path.join(phase, f"avg_{evaluation_key}_all_models_bar_chart.png"))

def plot_stacked_bar_for_phase(data, phase, runs, evaluation_key):
    model_evaluation = {}

    for run_name, run_data in data.items():
        for model in run_data:
            model_name = get_formatted_model_name(model)
            evaluation_value = model["evaluation_result"][evaluation_key]
            if model_name not in model_evaluation:
                model_evaluation[model_name] = []
            model_evaluation[model_name].append(evaluation_value)

    model_evaluation_sum = {model: sum(values) for model, values in model_evaluation.items()}
    sorted_models = sorted(model_evaluation_sum.keys(), key=lambda x: model_evaluation_sum[x], reverse=False)

    plot_data = []
    for model in sorted_models:
        for i, value in enumerate(model_evaluation[model]):
            plot_data.append({"Model": model, "Run": runs[i], evaluation_key: value})

    df = pd.DataFrame(plot_data)

    fig = plt_exp.bar(df, x="Model", y=evaluation_key, color="Run",
                      title=f"{evaluation_key} per Model (Stacked by Runs)",
                      labels={"Model": "Model", evaluation_key: evaluation_key, "Run": "Run"},
                      text_auto=True)

    fig.update_traces(textangle=-90, textposition="inside")
    fig.update_layout(xaxis_tickangle=-45, height=1200, width=2400)
    fig.write_image(os.path.join(phase, f"stacked_{evaluation_key}_bar_chart.png"))


def plot_multirun_phase_results(phase_data, phase_number):
    plot_stacked_bar_for_phase(phase_data, phase_number, RUNS_TO_TEST, "accuracy")
    plot_stacked_average_bar_for_phase(phase_data, phase_number, "accuracy")
    get_average_evaluation_grouped_by(phase_data, RUNS_TO_TEST, phase_number, "algebra", "accuracy")
    get_average_evaluation_grouped_by(phase_data, RUNS_TO_TEST, phase_number, "color_space", "accuracy")
    get_average_evaluation_grouped_by(phase_data, RUNS_TO_TEST, phase_number, "type", "accuracy")

phase1_data = read_phase_data(PHASE_1, RUNS_TO_TEST, only_hypercomplex=False)
phase2_data = read_phase_data(PHASE_2, RUNS_TO_TEST, only_hypercomplex=False)
phase3_data = read_phase_data(PHASE_3, RUNS_TO_TEST, only_hypercomplex=False)
phase4_data = read_phase_data(PHASE_4, RUNS_TO_TEST, only_hypercomplex=False)

plot_multirun_phase_results(phase1_data, PHASE_1)
plot_multirun_phase_results(phase2_data, PHASE_2)
plot_multirun_phase_results(phase3_data, PHASE_3)
plot_multirun_phase_results(phase4_data, PHASE_4)

# ------------------- Phase 1 vs Phase 2 -------------------
get_average_evaluation_between_phases_grouped_by(phase1_data, phase2_data, (1, 2), RUNS_TO_TEST, "color_space", "accuracy")

# ------------------- Phase 3 vs Phase 4 -------------------
get_average_evaluation_between_phases_grouped_by(phase3_data, phase4_data, (3, 4), RUNS_TO_TEST, "color_space", "accuracy")