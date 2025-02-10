import json, os
import plotly.express as plt_exp
import pandas as pd

# ------------------- Static parameters -------------------
run_dir = "run0"
json_result_file = "training.json"
# ---------------------------------------------------------

def get_formatted_model_name(model):
    base_name = f"{model['model_name']['type']}-{model['model_name']['color_space']}"
    if model['model_name']['algebra']:
        base_name += f"-{model['model_name']['algebra']}"
    return base_name

results_dir = os.path.join(run_dir, "results")

results_list = []

for single_result in os.listdir(results_dir):
    json_result_path = os.path.join(results_dir, single_result, json_result_file)
    with open(json_result_path, "r") as file:
        data = json.load(file)
        results_list.append(data)


model_name = [get_formatted_model_name(model) for model in results_list]
model_eval_accuracy = [model["evaluation_result"]["accuracy"] for model in results_list]
model_precision_accuracy = [model["evaluation_result"]["Precision"] for model in results_list]
model_recall_accuracy = [model["evaluation_result"]["Recall"] for model in results_list]
model_f1_score = [2 * (p * r) / (p + r) if (p + r) > 0 else 0 for p, r in zip(model_precision_accuracy, model_recall_accuracy)]
model_training_time = [model["training_time_seconds"] for model in results_list]

df = pd.DataFrame({
    "Model": model_name,
    "Accuracy": model_eval_accuracy,
    "F1 Score": model_f1_score,
    "Training Time": model_training_time
})

# -------------------------- Accuracy plot --------------------------
df = df.sort_values(by="Accuracy", ascending=True)

fig = plt_exp.bar(df, x="Model", y="Accuracy", title="Accuracy comparison",
             labels={"Model": "Model", "Accuracy": "Accuracy"},
             text_auto=True)

fig.update_layout(xaxis_tickangle=-45, height=600, width=1200)
fig.show()
fig.write_image(os.path.join(run_dir, "accuracy_bar_chart.png"))

# -------------------------- F1 Score plot --------------------------
df = df.sort_values(by="F1 Score", ascending=True)

fig = plt_exp.bar(df, x="Model", y="F1 Score", title="F1 Score comparison",
             labels={"Model": "Model", "F1 Score": "F1 Score"},
             text_auto=True)

fig.update_layout(xaxis_tickangle=-45, height=600, width=1200)
fig.show()
fig.write_image(os.path.join(run_dir, "f1_score_bar_chart.png"))

# -------------------------- Training time plot --------------------------
df = df.sort_values(by="Training Time", ascending=True)

fig = plt_exp.bar(df, x="Model", y="Training Time", title="Training time comparison",
             labels={"Model": "Model", "Training Time": "Training Time"},
             text_auto=True)

fig.update_layout(xaxis_tickangle=-45, height=600, width=1200)
fig.show()
fig.write_image(os.path.join(run_dir, "training_time_bar_chart.png"))