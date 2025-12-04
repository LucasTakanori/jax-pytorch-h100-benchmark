#!/usr/bin/env python3
"""
Verify benchmark completeness and generate comprehensive result tables
"""
import os
import glob
import pandas as pd
from collections import defaultdict

RESULTS_DIR = "../results"

def check_inference_completeness():
    """Check if we have all required inference configurations"""
    required_models = ['resnet50', 'vit_b_16', 'mobilenet_v3_small', 'efficientnet_b0']
    required_frameworks = ['pytorch', 'jax']
    required_batch_sizes = [1, 8, 32, 128]
    
    all_files = glob.glob(os.path.join(RESULTS_DIR, "inference", "*", "*.csv"))
    
    # Load all data
    all_data = []
    for f in all_files:
        try:
            df = pd.read_csv(f)
            all_data.append(df)
        except:
            continue
    
    if not all_data:
        print("❌ No inference data found!")
        return pd.DataFrame(), {}
    
    combined = pd.concat(all_data, ignore_index=True)
    
    # Check what we have
    found_configs = defaultdict(set)
    for _, row in combined.iterrows():
        key = (row['model'], row['framework'])
        found_configs[key].add(row['batch_size'])
    
    print("\n" + "="*60)
    print("INFERENCE BENCHMARK COMPLETENESS CHECK")
    print("="*60)
    
    missing = []
    for model in required_models:
        for framework in required_frameworks:
            key = (model, framework)
            found_bs = found_configs.get(key, set())
            missing_bs = set(required_batch_sizes) - found_bs
            
            status = "✓" if not missing_bs else "✗"
            print(f"{status} {framework:8s} {model:20s}: {sorted(found_bs)} ", end="")
            if missing_bs:
                print(f"MISSING: {sorted(missing_bs)}")
                missing.append((model, framework, missing_bs))
            else:
                print()
    
    print(f"\nTotal configurations: {len(found_configs)}/{len(required_models) * len(required_frameworks)}")
    print(f"Expected: {len(required_models)} models × {len(required_frameworks)} frameworks = {len(required_models) * len(required_frameworks)} configs")
    
    return combined, missing

def check_training_completeness():
    """Check training benchmark completeness"""
    required_models = ['resnet50', 'vit_b_16', 'mobilenet_v3_small', 'efficientnet_b0']
    required_batch_sizes = [32, 64, 128]
    
    all_files = glob.glob(os.path.join(RESULTS_DIR, "training", "*", "training_*.csv"))
    
    all_data = []
    for f in all_files:
        try:
            df = pd.read_csv(f)
            df['run_id'] = os.path.basename(f)
            all_data.append(df)
        except:
            continue
    
    if not all_data:
        print("\n❌ No training data found!")
        return pd.DataFrame(), {}
    
    combined = pd.concat(all_data, ignore_index=True)
    
    # Check what we have
    found_configs = defaultdict(set)
    for _, row in combined.iterrows():
        key = (row['model'], row['framework'])
        found_configs[key].add(row['batch_size'])
    
    print("\n" + "="*60)
    print("TRAINING BENCHMARK COMPLETENESS CHECK")
    print("="*60)
    
    missing = []
    for model in required_models:
        framework = 'pytorch'  # Only PyTorch training so far
        key = (model, framework)
        found_bs = found_configs.get(key, set())
        missing_bs = set(required_batch_sizes) - found_bs
        
        status = "✓" if not missing_bs else "✗"
        print(f"{status} {framework:8s} {model:20s}: {sorted(found_bs)} ", end="")
        if missing_bs:
            print(f"MISSING: {sorted(missing_bs)}")
            missing.append((model, framework, missing_bs))
        else:
            print()
    
    print(f"\nTotal configurations: {len(found_configs)}/{len(required_models)}")
    print(f"Expected: {len(required_models)} models × 1 framework (PyTorch) = {len(required_models)} configs")
    
    return combined, missing

def generate_full_inference_table(df):
    """Generate comprehensive inference results table"""
    if df.empty:
        return ""
    
    # Select relevant columns
    cols = ['framework', 'model', 'batch_size', 'latency_mean_ms', 'throughput_ips', 'memory_mb']
    table_df = df[cols].copy()
    
    # Escape underscores in model names for LaTeX
    table_df['model'] = table_df['model'].str.replace('_', '\\_')
    
    # Sort by framework, model, batch_size
    table_df = table_df.sort_values(['framework', 'model', 'batch_size'])
    
    # Format numbers
    table_df['latency_mean_ms'] = table_df['latency_mean_ms'].map('{:.2f}'.format)
    table_df['throughput_ips'] = table_df['throughput_ips'].map('{:.1f}'.format)
    table_df['memory_mb'] = table_df['memory_mb'].map('{:.0f}'.format)
    
    # Rename columns
    table_df.columns = ['Framework', 'Model', 'BS', 'Latency (ms)', 'Throughput (img/s)', 'Memory (MB)']
    
    latex = table_df.to_latex(index=False, 
                               caption="Complete Inference Results (All Configurations)",
                               label="tab:inference_full",
                               column_format='llrrrr',
                               escape=False)  # We already escaped manually
    
    # Add centering
    latex = latex.replace(r'\begin{tabular}', r'\centering' + '\n' + r'\begin{tabular}')
    
    with open("inference_full.tex", "w") as f:
        f.write(latex)
    
    print(f"\n✓ Generated inference_full.tex ({len(table_df)} rows)")
    return latex

def generate_full_training_table(df):
    """Generate comprehensive training results table"""
    if df.empty:
        return ""
    
    # Get final epoch for each run
    final_epochs = df.loc[df.groupby(['model', 'framework', 'batch_size', 'run_id'])['epoch'].idxmax()]
    
    # Aggregate by model, framework, batch_size
    agg_df = final_epochs.groupby(['model', 'framework', 'batch_size']).agg({
        'val_acc_top1': 'mean',
        'train_loss': 'mean',
        'epoch_duration_s': 'mean',
        'energy_j': 'mean'
    }).reset_index()
    
    # Escape underscores in model names for LaTeX
    agg_df['model'] = agg_df['model'].str.replace('_', '\\_')
    
    # Sort
    agg_df = agg_df.sort_values(['framework', 'model', 'batch_size'])
    
    # Format
    agg_df['val_acc_top1'] = agg_df['val_acc_top1'].map('{:.2f}'.format)
    agg_df['train_loss'] = agg_df['train_loss'].map('{:.4f}'.format)
    agg_df['epoch_duration_s'] = agg_df['epoch_duration_s'].map('{:.1f}'.format)
    agg_df['energy_j'] = agg_df['energy_j'].map('{:.1f}'.format)
    
    # Rename
    agg_df.columns = ['Model', 'Framework', 'BS', 'Val Acc (\\%)', 'Train Loss', 'Epoch Time (s)', 'Energy (J)']
    
    latex = agg_df.to_latex(index=False,
                            caption="Complete Training Results (Final Epoch, All Configurations)",
                            label="tab:training_full",
                            column_format='llrrrrr',
                            escape=False)  # We already escaped manually
    
    # Add centering
    latex = latex.replace(r'\begin{tabular}', r'\centering' + '\n' + r'\begin{tabular}')
    
    with open("training_full.tex", "w") as f:
        f.write(latex)
    
    print(f"✓ Generated training_full.tex ({len(agg_df)} rows)")
    return latex

def main():
    print("\n" + "="*60)
    print("BENCHMARK VERIFICATION AND TABLE GENERATION")
    print("="*60)
    
    # Check inference
    inf_df, inf_missing = check_inference_completeness()
    
    # Check training
    train_df, train_missing = check_training_completeness()
    
    # Generate tables
    print("\n" + "="*60)
    print("GENERATING COMPREHENSIVE TABLES")
    print("="*60)
    
    if not inf_df.empty:
        generate_full_inference_table(inf_df)
    
    if not train_df.empty:
        generate_full_training_table(train_df)
    
    # Summary
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    
    if inf_missing:
        print(f"\n⚠️  Missing {len(inf_missing)} inference configurations:")
        for model, framework, bs_list in inf_missing:
            print(f"   - {framework}/{model}: batch sizes {sorted(bs_list)}")
    else:
        print("\n✓ All inference configurations complete!")
    
    if train_missing:
        print(f"\n⚠️  Missing {len(train_missing)} training configurations:")
        for model, framework, bs_list in train_missing:
            print(f"   - {framework}/{model}: batch sizes {sorted(bs_list)}")
    else:
        print("\n✓ All training configurations complete!")
    
    print("\n" + "="*60)

if __name__ == "__main__":
    main()
