import os
import glob
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Set style
sns.set_theme(style="whitegrid")
plt.rcParams.update({'figure.max_open_warning': 0})

RESULTS_DIR = "../results"
OUTPUT_DIR = "."

def load_inference_data():
    all_files = glob.glob(os.path.join(RESULTS_DIR, "inference", "*", "*.csv"))
    df_list = []
    for filename in all_files:
        try:
            df = pd.read_csv(filename)
            df_list.append(df)
        except pd.errors.EmptyDataError:
            print(f"Skipping empty file: {filename}")
            continue
    
    if not df_list:
        return pd.DataFrame()
    
    combined_df = pd.concat(df_list, ignore_index=True)
    return combined_df

def load_training_data():
    all_files = glob.glob(os.path.join(RESULTS_DIR, "training", "*", "*.csv"))
    df_list = []
    for filename in all_files:
        try:
            df = pd.read_csv(filename)
            # Add a run_id based on filename to distinguish different runs
            df['run_id'] = os.path.basename(filename)
            df_list.append(df)
        except pd.errors.EmptyDataError:
            print(f"Skipping empty file: {filename}")
            continue

    if not df_list:
        return pd.DataFrame()
    
    combined_df = pd.concat(df_list, ignore_index=True)
    return combined_df

def plot_inference_metrics(df):
    models = df['model'].unique()
    
    for model in models:
        model_df = df[df['model'] == model]
        
        # Latency
        plt.figure(figsize=(10, 6))
        sns.lineplot(data=model_df, x='batch_size', y='latency_mean_ms', hue='framework', marker='o')
        plt.title(f'{model} Inference Latency vs Batch Size')
        plt.xlabel('Batch Size')
        plt.ylabel('Latency (ms)')
        plt.xscale('log', base=2)
        plt.yscale('log')
        plt.savefig(os.path.join(OUTPUT_DIR, f'{model}_latency.pdf'))
        plt.close()

        # Throughput
        plt.figure(figsize=(10, 6))
        sns.lineplot(data=model_df, x='batch_size', y='throughput_ips', hue='framework', marker='o')
        plt.title(f'{model} Inference Throughput vs Batch Size')
        plt.xlabel('Batch Size')
        plt.ylabel('Throughput (samples/sec)')
        plt.xscale('log', base=2)
        plt.savefig(os.path.join(OUTPUT_DIR, f'{model}_throughput.pdf'))
        plt.close()
        
        # Memory
        plt.figure(figsize=(10, 6))
        sns.lineplot(data=model_df, x='batch_size', y='memory_mb', hue='framework', marker='o')
        plt.title(f'{model} Peak Memory Usage vs Batch Size')
        plt.xlabel('Batch Size')
        plt.ylabel('Memory (MB)')
        plt.xscale('log', base=2)
        plt.savefig(os.path.join(OUTPUT_DIR, f'{model}_memory.pdf'))
        plt.close()

def plot_training_metrics(df):
    if df.empty:
        print("No training data found.")
        return

    models = df['model'].unique()
    for model in models:
        model_df = df[df['model'] == model]
        
        # Accuracy
        plt.figure(figsize=(10, 6))
        sns.lineplot(data=model_df, x='epoch', y='val_acc_top1', hue='framework', style='batch_size', marker='o')
        plt.title(f'{model} Training Validation Accuracy')
        plt.xlabel('Epoch')
        plt.ylabel('Top-1 Accuracy (%)')
        plt.savefig(os.path.join(OUTPUT_DIR, f'{model}_training_accuracy.pdf'))
        plt.close()
        
        # Loss
        plt.figure(figsize=(10, 6))
        sns.lineplot(data=model_df, x='epoch', y='train_loss', hue='framework', style='batch_size', marker='o')
        plt.title(f'{model} Training Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.savefig(os.path.join(OUTPUT_DIR, f'{model}_training_loss.pdf'))
        plt.close()

def generate_latex_tables(inf_df, train_df):
    # Inference Summary Table (Max Batch Size)
    max_bs_df = inf_df.loc[inf_df.groupby(['model', 'framework'])['batch_size'].idxmax()]
    
    # Select columns that exist
    base_cols = ['model', 'framework', 'batch_size', 'throughput_ips', 'latency_mean_ms', 'memory_mb']
    optional_cols = ['energy_j']
    
    cols_to_use = base_cols + [col for col in optional_cols if col in max_bs_df.columns]
    summary_table = max_bs_df[cols_to_use].copy()
    
    # Escape underscores in model names for LaTeX
    summary_table['model'] = summary_table['model'].str.replace('_', '\\_')
    
    # Create column names
    col_names = ['Model', 'Framework', 'Batch Size', 'Throughput (img/s)', 'Latency (ms)', 'Memory (MB)']
    if 'energy_j' in cols_to_use:
        col_names.append('Energy (J)')
    
    summary_table.columns = col_names
    
    # Format numbers
    summary_table['Throughput (img/s)'] = summary_table['Throughput (img/s)'].map('{:.1f}'.format)
    summary_table['Latency (ms)'] = summary_table['Latency (ms)'].map('{:.2f}'.format)
    summary_table['Memory (MB)'] = summary_table['Memory (MB)'].map('{:.0f}'.format)
    if 'Energy (J)' in summary_table.columns:
        summary_table['Energy (J)'] = summary_table['Energy (J)'].map('{:.2f}'.format)
    
    latex_code = summary_table.to_latex(index=False, caption="Inference Performance Summary (Max Batch Size)", label="tab:inference_summary", escape=False)
    
    # Add centering
    latex_code = latex_code.replace(r'\begin{tabular}', r'\centering' + '\n' + r'\begin{tabular}')
    
    with open(os.path.join(OUTPUT_DIR, "inference_summary.tex"), "w") as f:
        f.write(latex_code)

    # Training Summary Table (Last Epoch)
    if not train_df.empty:
        # Get last epoch for each run
        last_epoch_idx = train_df.groupby(['model', 'framework', 'batch_size', 'run_id'])['epoch'].idxmax()
        final_train_df = train_df.loc[last_epoch_idx]
        
        # Select columns that exist
        train_base_cols = ['val_acc_top1', 'epoch_duration_s']
        train_optional_cols = ['energy_j']
        
        train_cols_to_use = train_base_cols + [col for col in train_optional_cols if col in final_train_df.columns]
        
        # Aggregate over runs (mean)
        train_summary = final_train_df.groupby(['model', 'framework', 'batch_size'])[train_cols_to_use].mean().reset_index()
        
        # Escape underscores in model names for LaTeX
        train_summary['model'] = train_summary['model'].str.replace('_', '\\_')
        
        train_col_names = ['Model', 'Framework', 'Batch Size', 'Val Acc (\\%)', 'Epoch Time (s)']
        if 'energy_j' in train_cols_to_use:
            train_col_names.append('Energy/Epoch (J)')
        
        train_summary.columns = train_col_names
        
        train_summary['Val Acc (\\%)'] = train_summary['Val Acc (\\%)'].map('{:.2f}'.format)
        train_summary['Epoch Time (s)'] = train_summary['Epoch Time (s)'].map('{:.1f}'.format)
        if 'Energy/Epoch (J)' in train_summary.columns:
            train_summary['Energy/Epoch (J)'] = train_summary['Energy/Epoch (J)'].map('{:.1f}'.format)
        
        latex_code_train = train_summary.to_latex(index=False, caption="Training Performance Summary (Final Epoch)", label="tab:training_summary", escape=False)
        
        # Add centering
        latex_code_train = latex_code_train.replace(r'\begin{tabular}', r'\centering' + '\n' + r'\begin{tabular}')
        
        with open(os.path.join(OUTPUT_DIR, "training_summary.tex"), "w") as f:
            f.write(latex_code_train)

def main():
    print("Loading data...")
    inf_df = load_inference_data()
    train_df = load_training_data()
    
    print("Generating inference plots...")
    if not inf_df.empty:
        plot_inference_metrics(inf_df)
    
    print("Generating training plots...")
    if not train_df.empty:
        plot_training_metrics(train_df)
        
    print("Generating LaTeX tables...")
    generate_latex_tables(inf_df, train_df)
    
    print("Done!")

if __name__ == "__main__":
    main()
