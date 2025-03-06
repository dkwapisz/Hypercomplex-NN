# Generating charts for analysis

1. Create new `phase<num>` folder in the `analysis`, as well as `phase<num>/results_zip` directory.
2. Copy your results from `final_results` into `phase<num>/results_zip`.
3. Run the following command:
```bash
for i in {1..10}; do mkdir "run$i"; done
```
4. Run the following command:
```bash
for i in {1..10}; do unzip "results_zip/results_run$i.zip" -d "run$i"; done
```
5. Remove the `results_zip` directory as it is no longer needed.
6. Add your phases to the `SingleRunAnalysis.py` and `MultiRunAnalysis.py` scripts.
7. Run `SingleRunAnalysis.py` to generate charts for runs and `MultiRunAnalysis.py` to generate the charts for phases.

