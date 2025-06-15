import matplotlib
matplotlib.use('TkAgg')  # Set backend for macOS
import pandas as pd
import matplotlib.pyplot as plt
import argparse
import os
from datetime import datetime, timedelta

def parse_delay_data(filepath, sampling_interval_seconds=60):
    columns = ["Metric", "PacketCount", "Average", "Min", "Max",
               "75th%", "80th%", "85th%", "90th%", "95th%", "99th%", "999th%", "9999th%"]
    data = []
    first_occurrence = {'MDDELAY': None, 'TOTALDELAY': None, 'DELAYWITHORDER': None}

    with open(filepath, 'r') as f:
        for idx, line in enumerate(f):
            line = line.strip()
            if line.startswith(('MDDELAY', 'TOTALDELAY', 'DELAYWITHORDER')):
                parts = line.split()
                metric = parts[0]
                try:
                    row = [metric] + list(map(float, parts[1:]))
                    if len(row) == len(columns):
                        # Track first occurrence index
                        if first_occurrence[metric] is None:
                            first_occurrence[metric] = idx
                        data.append(row)
                    else:
                        print(f"Skipping line with incorrect columns in {filepath}: {line}")
                except ValueError:
                    print(f"Skipping malformed line in {filepath}: {line}")

    df = pd.DataFrame(data, columns=columns)
    if df.empty:
        print(f"Warning: No valid data parsed from {filepath}")
    else:
        print(f"\nParsed data from {filepath}:")
        print(df.head())
        print(f"First occurrence indices: {first_occurrence}")
    return df, first_occurrence

def compare_metrics(df1, df2, label1, label2, output_dir, first_occurrence1, first_occurrence2, sampling_interval_seconds):
    metrics = ['MDDELAY', 'TOTALDELAY', 'DELAYWITHORDER']

    # === AVERAGE DELAY COMPARISON ===
    comparison = {}
    for metric in metrics:
        avg1 = df1[df1['Metric'] == metric]['Average'].mean()
        avg2 = df2[df2['Metric'] == metric]['Average'].mean()
        comparison[metric] = {
            f'{label1}_Average': round(avg1, 2),
            f'{label2}_Average': round(avg2, 2),
            'Difference': round(avg2 - avg1, 2),
            'Percent Change': round(((avg2 - avg1) / avg1 * 100), 2) if avg1 != 0 else float('inf')
        }

    comparison_df = pd.DataFrame(comparison).T
    print("\n=== Average Delay Comparison ===")
    print(comparison_df)

    avg_csv_path = os.path.join(output_dir, f'average_comparison_{label1.lower()}_vs_{label2.lower()}.csv')
    comparison_df.to_csv(avg_csv_path)
    print(f"Average comparison saved to '{avg_csv_path}'.")

    # === PERCENTILE COMPARISON ===
    percentiles = ['75th%', '80th%', '85th%', '90th%', '95th%', '99th%', '999th%', '9999th%']
    rows = []

    for metric in metrics:
        for p in percentiles:
            if p in df1.columns and p in df2.columns:
                p1 = df1[df1['Metric'] == metric][p].mean()
                p2 = df2[df2['Metric'] == metric][p].mean()
                rows.append({
                    'Metric': metric,
                    'Percentile': p,
                    f'{label1}': round(p1, 2),
                    f'{label2}': round(p2, 2),
                    'Difference': round(p2 - p1, 2),
                    'Percent Change': round(((p2 - p1) / p1 * 100), 2) if p1 != 0 else float('inf')
                })
            else:
                print(f"Warning: {p} column not found for metric {metric}")

    percentile_comparison_df = pd.DataFrame(rows)
    print("\n=== Percentile-wise Delay Comparison ===")
    print(percentile_comparison_df)

    percentile_csv_path = os.path.join(output_dir, f'percentile_comparison_{label1.lower()}_vs_{label2.lower()}.csv')
    percentile_comparison_df.to_csv(percentile_csv_path, index=False)
    print(f"Percentile-wise analysis has been saved to '{percentile_csv_path}'.")

    # Plot Percentile Graphs
    for metric in metrics:
        df_metric = percentile_comparison_df[percentile_comparison_df['Metric'] == metric]
        plt.figure(figsize=(10, 6))
        plt.plot(df_metric['Percentile'], df_metric[label1], label=f'{label1} {metric}', marker='o', color='blue')
        plt.plot(df_metric['Percentile'], df_metric[label2], label=f'{label2} {metric}', marker='o', color='orange')
        plt.xlabel('Percentiles')
        plt.ylabel('Milliseconds')
        plt.title(f'{metric} Percentile-wise Comparison')
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plot_path = os.path.join(output_dir, f'{label1.lower()}_vs_{label2.lower()}_{metric.lower()}_percentiles.png')
        plt.savefig(plot_path)
        plt.show()

    # === INTRADAY AVERAGE COMPARISON ===
    base_time = datetime(2024, 1, 1, 8, 52)  # Start at 08:50 for earliest metric
    for metric in metrics:
        plt.figure(figsize=(10, 6))
        df1_metric = df1[df1['Metric'] == metric].copy()
        df2_metric = df2[df2['Metric'] == metric].copy()

        # Get first occurrence indices
        idx1 = first_occurrence1.get(metric, None)
        idx2 = first_occurrence2.get(metric, None)

        # Adjust timestamps based on first occurrence
        if not df1_metric.empty and idx1 is not None:
            start_time1 = base_time + timedelta(seconds=idx1 * sampling_interval_seconds)
            df1_metric['Timestamp'] = pd.date_range(start=start_time1, periods=len(df1_metric), freq=f'{sampling_interval_seconds}S')
            df1_metric['Smoothed'] = df1_metric['Average'].rolling(window=5, min_periods=1).mean()
            plt.plot(df1_metric['Timestamp'], df1_metric['Smoothed'], label=f'{label1} {metric}', color='blue')
            print(f"{label1} {metric}: First occurrence at index {idx1}, starts at {start_time1.strftime('%H:%M')}")
        else:
            print(f"Warning: No data for {metric} in {label1} or metric not found")

        if not df2_metric.empty and idx2 is not None:
            start_time2 = base_time + timedelta(seconds=idx2 * sampling_interval_seconds)
            df2_metric['Timestamp'] = pd.date_range(start=start_time2, periods=len(df2_metric), freq=f'{sampling_interval_seconds}S')
            df2_metric['Smoothed'] = df2_metric['Average'].rolling(window=5, min_periods=1).mean()
            plt.plot(df2_metric['Timestamp'], df2_metric['Smoothed'], label=f'{label2} {metric}', color='orange')
            print(f"{label2} {metric}: First occurrence at index {idx2}, starts at {start_time2.strftime('%H:%M')}")
        else:
            print(f"Warning: No data for {metric} in {label2} or metric not found")

        plt.xlabel('Time')
        plt.ylabel('Smoothed Avg Delay (ms)')
        plt.title(f'{metric} Intraday Smoothed Comparison')
        plt.legend()
        plt.grid(True)
        plt.gca().xaxis.set_major_formatter(plt.matplotlib.dates.DateFormatter('%H:%M'))
        plt.xticks(rotation=45)
        plt.tight_layout()
        plot_path = os.path.join(output_dir, f'{label1.lower()}_vs_{label2.lower()}_{metric.lower()}_intraday_smoothed.png')
        plt.savefig(plot_path)
        print(f"Saved intraday plot: {plot_path}")
        plt.show()

def main():
    parser = argparse.ArgumentParser(description="Compare latency data between two machines")
    parser.add_argument('file1', help="Path to the first machine's latency file")
    parser.add_argument('file2', help="Path to the second machine's latency file")
    parser.add_argument('--label1', default="Machine_A", help="Label for the first machine")
    parser.add_argument('--label2', default="Machine_B", help="Label for the second machine")
    parser.add_argument('--interval', type=int, default=60, help="Sampling interval in seconds (default: 60)")
    args = parser.parse_args()

    if not os.path.isfile(args.file1) or not os.path.isfile(args.file2):
        print("Error: One or both input files do not exist.")
        return

    label1 = args.label1 if args.label1 != "Machine_A" else os.path.splitext(os.path.basename(args.file1))[0]
    label2 = args.label2 if args.label2 != "Machine_B" else os.path.splitext(os.path.basename(args.file2))[0]

    df1, first_occurrence1 = parse_delay_data(args.file1, args.interval)
    df2, first_occurrence2 = parse_delay_data(args.file2, args.interval)

    output_dir = os.path.dirname(args.file1) or '.'
    compare_metrics(df1, df2, label1, label2, output_dir, first_occurrence1, first_occurrence2, args.interval)

if __name__ == '__main__':
    main()