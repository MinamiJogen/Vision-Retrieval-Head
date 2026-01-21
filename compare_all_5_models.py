#!/usr/bin/env python3
"""
Compare results from all 5 evaluated models
- 3 LongVA models
- 1 InternVL model
- 1 Qwen3 model

Usage: python compare_all_5_models.py
"""

import pandas as pd
import glob
import os
import sys

def main():
    # Model groups
    models_info = {
        'LongVA': ['LongVA-Temporal-v1', 'LongVA-Temporal-v2', 'LongVA-7B'],
        'InternVL': ['InternVL3_5-8B'],
        'Qwen3': ['Qwen3-VL-8B-Instruct']
    }

    # Find result directories
    longva_dirs = sorted(glob.glob('longva_results_*'))
    internvl_dirs = sorted(glob.glob('internvl_results_*'))
    qwen3_dirs = sorted(glob.glob('qwen3_results_*'))
    # Also check old combined directory
    qwen_internvl_dirs = sorted(glob.glob('qwen_internvl_results_*'))

    if not longva_dirs and not internvl_dirs and not qwen3_dirs and not qwen_internvl_dirs:
        print("❌ No results found!")
        print("Make sure you run this script from VLMEvalKit directory")
        sys.exit(1)

    longva_dir = longva_dirs[-1] if longva_dirs else None
    internvl_dir = internvl_dirs[-1] if internvl_dirs else None
    qwen3_dir = qwen3_dirs[-1] if qwen3_dirs else None
    qwen_internvl_dir = qwen_internvl_dirs[-1] if qwen_internvl_dirs else None

    print("="*80)
    print("Complete Model Comparison (5 Models)")
    print("="*80)

    if longva_dir:
        print(f"📁 LongVA results: {longva_dir}")
    if internvl_dir:
        print(f"📁 InternVL results: {internvl_dir}")
    if qwen3_dir:
        print(f"📁 Qwen3 results: {qwen3_dir}")
    if qwen_internvl_dir:
        print(f"📁 Qwen/InternVL (old) results: {qwen_internvl_dir}")

    print("")

    # Datasets (removed MMMU_DEV_VAL due to compatibility issues)
    datasets = [
        'MMBench_DEV_EN', 'MME', 'SEEDBench_IMG',
        'HallusionBench', 'AI2D_TEST', 'OCRBench', 'MathVista_MINI',
        'RealWorldQA', 'POPE'
    ]

    # Collect all results
    all_results = {}

    for dataset in datasets:
        all_results[dataset] = {}

        # LongVA models
        if longva_dir:
            for model in models_info['LongVA']:
                csv_file = f"{longva_dir}/{model}/{model}_{dataset}.csv"
                if os.path.exists(csv_file):
                    try:
                        df = pd.read_csv(csv_file)
                        if len(df) > 0 and len(df.columns) > 1:
                            score = df.iloc[0, 1]
                            all_results[dataset][model] = score
                        else:
                            all_results[dataset][model] = "N/A"
                    except Exception as e:
                        all_results[dataset][model] = "Error"
                else:
                    all_results[dataset][model] = "Not completed"

        # InternVL models - check new separate directory first
        for model in models_info['InternVL']:
            found = False
            if internvl_dir:
                csv_file = f"{internvl_dir}/{model}/{model}_{dataset}.csv"
                if os.path.exists(csv_file):
                    found = True
                    try:
                        df = pd.read_csv(csv_file)
                        if len(df) > 0 and len(df.columns) > 1:
                            score = df.iloc[0, 1]
                            all_results[dataset][model] = score
                        else:
                            all_results[dataset][model] = "N/A"
                    except Exception as e:
                        all_results[dataset][model] = "Error"

            # Fallback to old combined directory
            if not found and qwen_internvl_dir:
                csv_file = f"{qwen_internvl_dir}/{model}/{model}_{dataset}.csv"
                if os.path.exists(csv_file):
                    try:
                        df = pd.read_csv(csv_file)
                        if len(df) > 0 and len(df.columns) > 1:
                            score = df.iloc[0, 1]
                            all_results[dataset][model] = score
                        else:
                            all_results[dataset][model] = "N/A"
                    except Exception as e:
                        all_results[dataset][model] = "Error"
                else:
                    all_results[dataset][model] = "Not completed"
            elif not found:
                all_results[dataset][model] = "Not completed"

        # Qwen3 models - check new separate directory first
        for model in models_info['Qwen3']:
            found = False
            if qwen3_dir:
                csv_file = f"{qwen3_dir}/{model}/{model}_{dataset}.csv"
                if os.path.exists(csv_file):
                    found = True
                    try:
                        df = pd.read_csv(csv_file)
                        if len(df) > 0 and len(df.columns) > 1:
                            score = df.iloc[0, 1]
                            all_results[dataset][model] = score
                        else:
                            all_results[dataset][model] = "N/A"
                    except Exception as e:
                        all_results[dataset][model] = "Error"

            # Fallback to old combined directory
            if not found and qwen_internvl_dir:
                csv_file = f"{qwen_internvl_dir}/{model}/{model}_{dataset}.csv"
                if os.path.exists(csv_file):
                    try:
                        df = pd.read_csv(csv_file)
                        if len(df) > 0 and len(df.columns) > 1:
                            score = df.iloc[0, 1]
                            all_results[dataset][model] = score
                        else:
                            all_results[dataset][model] = "N/A"
                    except Exception as e:
                        all_results[dataset][model] = "Error"
                else:
                    all_results[dataset][model] = "Not completed"
            elif not found:
                all_results[dataset][model] = "Not completed"

    # Print detailed results
    print("\n" + "="*80)
    print("Detailed Results by Dataset")
    print("="*80)

    all_models = models_info['LongVA'] + models_info['InternVL'] + models_info['Qwen3']

    for dataset in datasets:
        print(f"\n📊 {dataset}:")
        print("-" * 80)

        for model in all_models:
            if model in all_results[dataset]:
                score = all_results[dataset][model]
                print(f"  {model:30s}: {score}")
        print("-" * 80)

    # Print summary table
    print("\n" + "="*80)
    print("Summary Table")
    print("="*80)
    print()

    # Header
    header = "Dataset".ljust(20)
    for model in all_models:
        header += model[:15].ljust(17)
    print(header)
    print("="*len(header))

    # Data rows
    for dataset in datasets:
        row = dataset[:19].ljust(20)
        for model in all_models:
            if model in all_results[dataset]:
                score_str = str(all_results[dataset][model])
                if score_str == "Not completed":
                    score_str = "-"
                elif score_str == "Error":
                    score_str = "ERR"
                row += score_str[:15].ljust(17)
            else:
                row += "-".ljust(17)
        print(row)

    print("="*len(header))
    print()

    # Statistics
    print("="*80)
    print("Completion Statistics")
    print("="*80)

    for model in all_models:
        completed = sum(1 for d in datasets if model in all_results[d] and
                       isinstance(all_results[d][model], (int, float)))
        total = len(datasets)
        percentage = (completed / total) * 100 if total > 0 else 0

        status = "✓" if completed == total else "⚠️"
        print(f"{status} {model:30s}: {completed}/{total} ({percentage:.0f}%)")

    print("="*80)
    print()

    # Export to CSV
    try:
        # Create a comprehensive results DataFrame
        df_data = []
        for dataset in datasets:
            row = {'Dataset': dataset}
            for model in all_models:
                if model in all_results[dataset]:
                    row[model] = all_results[dataset][model]
                else:
                    row[model] = None
            df_data.append(row)

        df_summary = pd.DataFrame(df_data)
        csv_filename = 'all_5_models_comparison.csv'
        df_summary.to_csv(csv_filename, index=False)
        print(f"📄 Results exported to: {csv_filename}")
        print()
    except Exception as e:
        print(f"⚠️  Could not export CSV: {e}")
        print()

if __name__ == "__main__":
    main()
