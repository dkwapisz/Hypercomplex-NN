import json
import os
from collections import defaultdict
import matplotlib.colors as mcolors
import numpy as np
import plotly.colors
import plotly.graph_objects as go


def read_phase_data_statistic(run_phase, runs, only_hypercomplex=False):
    iterations = ["iteration1", "iteration2", "iteration3", "iteration4", "iteration5"]
    data = {iteration: {run: [] for run in runs} for iteration in iterations}

    for iteration in iterations:
        for run in runs:
            trained_models = os.listdir(os.path.join(run_phase, iteration, run, "results"))
            for model in trained_models:
                file_path = os.path.join(run_phase, iteration, run, "results", model, "training.json")
                if os.path.exists(file_path):
                    with open(file_path, 'r') as f:
                        model_data = json.load(f)
                        if only_hypercomplex:
                            if model_data.get("model_name").get("type") == "HyperComplex":
                                data[iteration][run].append(model_data)
                        else:
                            data[iteration][run].append(model_data)

    return data


def evaluation_by(data, runs, evaluation_key):
    grouped_evaluation = {run: {} for run in runs}

    for run_name, run in data.items():
        for model in run:
            evaluation_value = model["evaluation_result"][evaluation_key]
            group_by_value = model["model_name"]["type"] + "-" + model["model_name"]["color_space"]

            if model["model_name"]["algebra"] != "":
                group_by_value += "-" + model["model_name"]["algebra"]

            if group_by_value not in grouped_evaluation[run_name]:
                grouped_evaluation[run_name][group_by_value] = []
            grouped_evaluation[run_name][group_by_value].append(evaluation_value)

    return grouped_evaluation


def get_average_and_std_evaluation_by_model(data, runs, run_phase, run_labels, evaluation_key="accuracy"):
    combined_stats = defaultdict(lambda: defaultdict(list))

    for iteration_data in data.values():
        for run_name, run_data in iteration_data.items():
            for model in run_data:
                evaluation_value = model["evaluation_result"][evaluation_key]
                group_by_value = model["model_name"]["type"] + "-" + model["model_name"]["color_space"]
                if model["model_name"]["algebra"] != "":
                    group_by_value += "-" + model["model_name"]["algebra"]
                combined_stats[group_by_value][run_name].append(evaluation_value)

    fig = go.Figure()

    colors = plotly.colors.qualitative.Plotly
    model_names = list(combined_stats.keys())
    color_map = {model: colors[i % len(colors)] for i, model in enumerate(model_names)}

    for model_name in model_names:
        run_values = combined_stats[model_name]

        x = []
        y = []
        y_upper = []
        y_lower = []

        for run_name in runs:
            if run_name in run_values:
                run_label = run_labels[runs.index(run_name)]
                values = run_values[run_name]
                avg = np.mean(values)
                std = np.std(values, ddof=1)

                x.append(run_label)
                y.append(avg)
                y_upper.append(avg + std)
                y_lower.append(avg - std)

        base_color = color_map[model_name]
        rgb = mcolors.to_rgb(base_color)
        r, g, b = [int(255 * x) for x in rgb]
        fill_color = f'rgba({r}, {g}, {b}, 0.3)'

        fig.add_trace(go.Scatter(
            x=x + x[::-1],
            y=y_upper + y_lower[::-1],
            fill='toself',
            fillcolor=fill_color,
            line=dict(color='rgba(255,255,255,0)'),
            hoverinfo="skip",
            showlegend=False,
            name=model_name + " std",
        ))

        fig.add_trace(go.Scatter(
            x=x,
            y=y,
            mode='lines+markers',
            name=model_name,
            line=dict(width=2, color=base_color),))

    fig.update_layout(
        title=f"Average {evaluation_key} for each model - HCNN with 4-times less parameters than CNN",
        xaxis=dict(
            tickmode='array',
            tickvals=RUN_LABELS_SPLIT_5_PLUS,
            ticktext=[str(v) for v in RUN_LABELS_SPLIT_5_PLUS],
            range=[0, 70]
        ),
        xaxis_title="Percentage of data used for training",
        yaxis_title=f"Average {evaluation_key}",
        height=800,
        width=1200
    )

    fig.write_image(os.path.join(run_phase, f"avg_std_{evaluation_key.lower()}_by_model_shaded_plot.png"))

def compare_hypercomplex_models_between_phases_shaded(phase1_path, phase2_path, runs, run_labels, evaluation_key="accuracy"):
    data_phase1 = read_phase_data_statistic(phase1_path, runs, only_hypercomplex=True)
    data_phase2 = read_phase_data_statistic(phase2_path, runs, only_hypercomplex=True)

    combined_data = {
        f"{"HCNN/4"}": data_phase1,
        f"{"HCNN/2"}": data_phase2
    }

    combined_stats = defaultdict(lambda: defaultdict(list))

    for phase_name, phase_data in combined_data.items():
        for iteration_data in phase_data.values():
            for run_name, run_data in iteration_data.items():
                for model in run_data:
                    model_name = model["model_name"]
                    if model_name["type"] != "HyperComplex":
                        continue
                    eval_val = model["evaluation_result"][evaluation_key]
                    group_key = f'{model_name["type"]}-{model_name["color_space"]}'
                    if model_name["algebra"]:
                        group_key += f'-{model_name["algebra"]}'
                    group_key += f" ({phase_name})"
                    combined_stats[group_key][run_name].append(eval_val)

    fig = go.Figure()
    colors = plotly.colors.qualitative.Plotly
    model_names = list(combined_stats.keys())
    color_map = {model: colors[i % len(colors)] for i, model in enumerate(model_names)}

    for model_name in model_names:
        run_values = combined_stats[model_name]
        x, y, y_upper, y_lower = [], [], [], []

        for run_name in runs:
            if run_name in run_values:
                run_label = run_labels[runs.index(run_name)]
                values = run_values[run_name]
                avg = np.mean(values)
                std = np.std(values, ddof=1)

                x.append(run_label)
                y.append(avg)
                y_upper.append(avg + std)
                y_lower.append(avg - std)

        base_color = color_map[model_name]
        rgb = mcolors.to_rgb(base_color)
        r, g, b = [int(255 * c) for c in rgb]
        fill_color = f'rgba({r}, {g}, {b}, 0.2)'

        fig.add_trace(go.Scatter(
            x=x + x[::-1],
            y=y_upper + y_lower[::-1],
            fill='toself',
            fillcolor=fill_color,
            line=dict(color='rgba(255,255,255,0)'),
            hoverinfo="skip",
            showlegend=False,
            name=model_name + " std",
        ))

        fig.add_trace(go.Scatter(
            x=x,
            y=y,
            mode='lines+markers',
            name=model_name,
            line=dict(width=2, color=base_color),
        ))

    fig.update_layout(
        title=f"Comparison of HyperComplex Models: {phase1_path} vs {phase2_path}",
        xaxis=dict(
            tickmode='array',
            tickvals=run_labels,
            ticktext=[str(v) for v in run_labels],
            range=[0, 70]
        ),
        xaxis_title="Percentage of data used for training",
        yaxis_title=f"Average {evaluation_key}",
        height=800,
        width=1200
    )

    fig.write_image(f"between_phases_results/comparison_{phase1_path}_vs_{phase2_path}_{evaluation_key}.png")

def compare_hypercomplex_models_between_phases_errorbars(phase1_path, phase2_path, runs, run_labels, evaluation_key="accuracy"):
    data_phase1 = read_phase_data_statistic(phase1_path, runs, only_hypercomplex=True)
    data_phase2 = read_phase_data_statistic(phase2_path, runs, only_hypercomplex=True)

    combined_data = {
        f"{"HCNN/4"}": data_phase1,
        f"{"HCNN/2"}": data_phase2
    }

    combined_stats = defaultdict(lambda: defaultdict(list))

    for phase_name, phase_data in combined_data.items():
        for iteration_data in phase_data.values():
            for run_name, run_data in iteration_data.items():
                for model in run_data:
                    model_name = model["model_name"]
                    if model_name["type"] != "HyperComplex":
                        continue
                    eval_val = model["evaluation_result"][evaluation_key]
                    group_key = f'{model_name["type"]}-{model_name["color_space"]}'
                    if model_name["algebra"]:
                        group_key += f'-{model_name["algebra"]}'
                    group_key += f" ({phase_name})"
                    combined_stats[group_key][run_name].append(eval_val)

    fig = go.Figure()
    colors = plotly.colors.qualitative.Plotly
    model_names = list(combined_stats.keys())
    color_map = {model: colors[i % len(colors)] for i, model in enumerate(model_names)}

    for model_name in model_names:
        run_values = combined_stats[model_name]
        x, y, error_y = [], [], []

        for run_name in runs:
            if run_name in run_values:
                run_label = run_labels[runs.index(run_name)]
                values = run_values[run_name]
                avg = np.mean(values)
                std = np.std(values, ddof=1)

                x.append(run_label)
                y.append(avg)
                error_y.append(std)

        fig.add_trace(go.Scatter(
            x=x,
            y=y,
            mode='lines+markers',
            name=model_name,
            line=dict(width=2, color=color_map[model_name]),
            error_y=dict(
                type='data',
                array=error_y,
                visible=True,
                thickness=1.5,
                width=6
            )
        ))

    fig.update_layout(
        title=f"Comparison of HyperComplex Models: {phase1_path} vs {phase2_path}",
        xaxis=dict(
            tickmode='array',
            tickvals=run_labels,
            ticktext=[str(v) for v in run_labels],
            range=[0, 70]
        ),
        xaxis_title="Percentage of data used for training",
        yaxis_title=f"Average {evaluation_key}",
        height=800,
        width=1200
    )

    fig.write_image(f"between_phases_results/comparison_{phase1_path}_vs_{phase2_path}_{evaluation_key}_errorbars.png")


RUN_LABELS_SPLIT_5_PLUS = [1, 3, 5, 10, 20, 30, 40, 50, 60, 70]
RUNS = ["run1", "run2", "run3", "run4", "run5", "run6", "run7", "run8", "run9", "run10"]

phase_data = read_phase_data_statistic("phase26-h4", RUNS, only_hypercomplex=False)
get_average_and_std_evaluation_by_model(phase_data, RUNS, "phase26-h4", RUN_LABELS_SPLIT_5_PLUS, "accuracy")

phase_data = read_phase_data_statistic("phase27-h2", RUNS, only_hypercomplex=False)
get_average_and_std_evaluation_by_model(phase_data, RUNS, "phase27-h2", RUN_LABELS_SPLIT_5_PLUS, "accuracy")

compare_hypercomplex_models_between_phases_shaded("phase26-h4", "phase27-h2", RUNS, RUN_LABELS_SPLIT_5_PLUS, "accuracy")
compare_hypercomplex_models_between_phases_errorbars("phase26-h4", "phase27-h2", RUNS, RUN_LABELS_SPLIT_5_PLUS, "accuracy")