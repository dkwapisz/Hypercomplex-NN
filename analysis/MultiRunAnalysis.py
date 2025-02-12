import json, os
import plotly.express as plt_exp
import pandas as pd

# ------------------- Static parameters -------------------
run_phase = "phase_1"
runs = ["run0"]
json_result_file = "training.json"


# ---------------------------------------------------------

def get_formatted_model_name(model):
    base_name = f"{model['model_name']['type']}-{model['model_name']['color_space']}"
    if model['model_name']['algebra']:
        base_name += f"-{model['model_name']['algebra']}"
    return base_name


results_list = []

for run in runs:
    results_dir = os.path.join(run_phase, run, "results")
    for single_result in os.listdir(results_dir):
        json_result_path = os.path.join(results_dir, single_result, json_result_file)
        with open(json_result_path, "r") as file:
            data = json.load(file)
            data['run'] = run
            results_list.append(data)

model_name = [get_formatted_model_name(model) for model in results_list]
run_name = [model['run'] for model in results_list]
model_eval_accuracy = [model["evaluation_result"]["accuracy"] for model in results_list]
model_precision_accuracy = [model["evaluation_result"]["Precision"] for model in results_list]
model_recall_accuracy = [model["evaluation_result"]["Recall"] for model in results_list]
model_f1_score = [2 * (p * r) / (p + r) if (p + r) > 0 else 0 for p, r in
                  zip(model_precision_accuracy, model_recall_accuracy)]
model_training_time = [model["training_time_seconds"] for model in results_list]

df = pd.DataFrame({
    "Model": model_name,
    "Run": run_name,
    "Accuracy": model_eval_accuracy,
    "F1 Score": model_f1_score,
    "Training Time": model_training_time
})

# TODO Fix to stop grouping by color, but keep coloring to mark Run

# -------------------------- Accuracy comparison plot --------------------------
df_best_accuracy = df.loc[df.groupby('Model')['Accuracy'].idxmax()]
df_best_accuracy_sorted = df_best_accuracy.sort_values(by="Accuracy", ascending=False)
fig_accuracy = plt_exp.bar(df_best_accuracy_sorted, x="Model", y="Accuracy", color="Run",
                           title="Best Accuracy Across Runs",
                           labels={"Model": "Model", "Accuracy": "Accuracy"},
                           text_auto=True)

fig_accuracy.update_layout(xaxis_tickangle=-45, height=600, width=1200)
fig_accuracy.show()
fig_accuracy.write_image(os.path.join(run_phase, "comparison", "best_accuracy_comparison.png"))

# -------------------------- F1 Score comparison plot --------------------------
df_best_f1 = df.loc[df.groupby('Model')['F1 Score'].idxmax()]
df_best_f1_sorted = df_best_f1.sort_values(by="F1 Score", ascending=False)

fig_f1 = plt_exp.bar(df_best_f1_sorted, x="Model", y="F1 Score", color="Run", title="Best F1 Score Across Runs",
                     labels={"Model": "Model", "F1 Score": "F1 Score"},
                     text_auto=True)

fig_f1.update_layout(xaxis_tickangle=-45, height=600, width=1200)
fig_f1.show()
fig_f1.write_image(os.path.join(run_phase, "comparison", "best_f1_score_comparison.png"))

# -------------------------- Training Time comparison plot --------------------------
df_best_training_time = df.loc[df.groupby('Model')['Training Time'].idxmin()]
df_best_training_time_sorted = df_best_training_time.sort_values(by="Training Time", ascending=True)

fig_training_time = plt_exp.bar(df_best_training_time_sorted, x="Model", y="Training Time", color="Run",
                                title="Best Training Time Across Runs",
                                labels={"Model": "Model", "Training Time": "Training Time (seconds)"},
                                text_auto=True)

fig_training_time.update_layout(xaxis_tickangle=-45, height=600, width=1200)
fig_training_time.show()
fig_training_time.write_image(os.path.join(run_phase, "comparison", "best_training_time_comparison.png"))
