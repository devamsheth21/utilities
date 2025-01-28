import pandas as pd
import numpy as np
import argparse
from scipy import stats
import logging

logging.basicConfig(level=logging.INFO)

def load_data(path):
    try:
        data = pd.read_csv(path)
        return data
    except Exception as e:
        logging.error(f"Error loading data: {e}")
        raise

def write_data(data, path):
    try:
        with open(path, 'w') as file:
            for dataset_name, metrics in data.items():
                file.write(f"Dataset: {dataset_name}\n")
                for key, value in metrics.items():
                    file.write(f"{key}: {value}\n")
                file.write("\n")
    except Exception as e:
        logging.error(f"Error writing data: {e}")
        raise

def calculate_confidence_interval(data, confidence=0.95):
    n = len(data)
    mean = np.mean(data)
    sem = stats.sem(data)
    margin = sem * stats.t.ppf((1 + confidence) / 2., n-1)
    return mean, margin

def boot_format(arr):
    mu = np.mean(arr)
    std = np.std(arr) 
    return f"{mu:0.3f}({mu-2*std:0.3f},{mu+2*std:0.3f})"
    
def get_metrics(df, dataset_name, metric_columns):
    df = df[df['dataset_name'] == dataset_name]
    metrics = {name: [] for name in metric_columns}    
    np.random.seed(123)
    
    for i in range(100): 
        sub_pop = np.random.randint(26, 75) / 100
        t_df = df.sample(frac=sub_pop, random_state=i)
        for name in metric_columns:
            metric_value = t_df[name].mean()  # Example metric calculation
            metrics[name].append(metric_value)
    
    out_d = {}
    for name, values in metrics.items():
        boot_ci = boot_format(values)
        mean, margin = calculate_confidence_interval(values)
        ci = f"{mean:0.3f}({mean-margin:0.3f},{mean+margin:0.3f})"
        out_d[name] = {'Bootstrap CI': boot_ci, 'Confidence Interval': ci}
    
    return out_d

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', type=str, required=True, help="Path to the CSV file containing dataset")
    parser.add_argument('--output_path', type=str, required=True, help="Path to save the output results")
    parser.add_argument('--dataset_column', type=str, required=True, help="Column name for dataset identification")
    parser.add_argument('--metric_columns', type=str, nargs='+', required=True, help="Column names for metrics")
    args = parser.parse_args()
    
    df = load_data(args.data_path)
    dataset_names = df[args.dataset_column].unique()
    
    all_metrics = {}
    for dataset_name in dataset_names:
        logging.info(f"Processing dataset: {dataset_name}")
        all_metrics[dataset_name] = get_metrics(df, dataset_name, args.metric_columns)
    
    write_data(all_metrics, args.output_path)

if __name__ == "__main__":
    main()