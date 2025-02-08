import json, os
import plotly.express as plt_exp
import pandas as pd

# ------------------- Static parameters -------------------
run_dir = "run1"
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
model_training_time = [model["training_time_seconds"] for model in results_list]

# -------------------------- Accuracy plot --------------------------
df = pd.DataFrame({"Name": model_name, "Value": model_eval_accuracy}).sort_values(by="Value", ascending=True)

fig = plt_exp.bar(df, x="Name", y="Value", title="Accuracy comparison",
             labels={"Name": "Model", "Value": "Accuracy"},
             text_auto=True)

fig.update_layout(xaxis_tickangle=-45, height=600, width=1200)
fig.show()
fig.write_image(os.path.join(run_dir, "accuracy_bar_chart.png"))

# -------------------------- Training time plot --------------------------
df = pd.DataFrame({"Name": model_name, "Value": model_training_time}).sort_values(by="Value", ascending=True)

fig = plt_exp.bar(df, x="Name", y="Value", title="Training time comparison",
             labels={"Name": "Model", "Value": "Training Time"},
             text_auto=True)

fig.update_layout(xaxis_tickangle=-45, height=600, width=1200)
fig.show()
fig.write_image(os.path.join(run_dir, "training_time_bar_chart.png"))