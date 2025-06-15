import pandas as pd
import matplotlib.pyplot as plt
import argparse
import os
from datetime import datetime, timedelta

def parse_delay_data(filepath):
    columns = ["Timestamp", "Metric", "PacketCount", "Average", "Min", "Max",
               "75th%", "80th%", "85th%", "90th%", "95th%", "99th%", "999th%", "9999th%"]
    data = []
    current_time = None

    with open(filepath, 'r') as f:
        for line in f:
            line = line.strip()
            if ':' in line and len(line.split()) == 1:
                try:
                    if line.count(':') == 1 and len(line.split(':')[1]) == 1:
                        line = line.replace(':', ':0')
                    current_time = datetime.strptime(line, "%H:%M")
                except ValueError:
                    print(f"Skipping malformed timestamp: {line}")
                    continue
            
            if line.startswith(('MDDELAY', 'TOTALDELAY', 'DELAYWITHORDER')) and current_time is not None:
                parts = line.split()
                metric = parts[0]
                try:
                    row = [current_time] + [metric] + list(map(float, parts[1:]))
                    if len(row) == len(columns):
                        data.append(row)
                except ValueError:
                    print(f"Skipping malformed line: {line}")
    
    return pd.DataFrame(data, columns=columns)

def align_time_series(df1, df2):
    start_time = max(df1['Timestamp'].min(), df2['Timestamp'].min())
    end_time = min(df1['Timestamp'].max(), df2['Timestamp'].max())
    df1_aligned = df1[(df1['Timestamp'] >= start_time) & (df1['Timestamp'] <= end_time)]
    df2_aligned = df2[(df2['Timestamp'] >= start_time) & (df2['Timestamp'] <= end_time)]
    return df1_aligned, df2_aligned

def compare_metrics(df1, df2, label1, label2, output_dir):
    metrics = ['MDDELAY', 'TOTALDELAY', 'DELAYWITHORDER']

    # Align for comparison metrics only
    df1_aligned, df2_aligned = align_time_series(df1, df2)
    
    if df1_aligned.empty or df2_aligned.empty:
        print("Warning: No overlapping time periods between the two datasets")
        return

    # Average Delay Comparison (CSV only)
    comparison = {}
    for metric in metrics:
        avg1 = df1_aligned[df1_aligned['Metric'] == metric]['Average'].mean()
        avg2 = df2_aligned[df2_aligned['Metric'] == metric]['Average'].mean()
        comparison[metric] = {
            f'{label1}_Average': avg1,
            f'{label2}_Average': avg2,
            'Difference': avg2 - avg1,
            'Percent Change': ((avg2 - avg1) / avg1 * 100) if avg1 != 0 else float('inf')
        }

    comparison_df = pd.DataFrame(comparison).T
    avg_csv_path = os.path.join(output_dir, 'average_comparison.csv')
    comparison_df.to_csv(avg_csv_path)
    print(f"Average delay comparison saved to '{avg_csv_path}'.")

    # Percentile Comparison (CSV + plots)
    percentiles = ['75th%', '80th%', '85th%', '90th%', '95th%', '99th%', '999th%', '9999th%']
    rows = []

    for metric in metrics:
        for p in percentiles:
            if p in df1.columns and p in df2.columns:
                p1 = df1_aligned[df1_aligned['Metric'] == metric][p].mean()
                p2 = df2_aligned[df2_aligned['Metric'] == metric][p].mean()
                rows.append({
                    'Metric': metric,
                    'Percentile': p,
                    f'{label1}': round(p1, 2),
                    f'{label2}': round(p2, 2),
                    'Difference': round(p2 - p1, 2),
                    'Percent Change': round(((p2 - p1) / p1 * 100), 2) if p1 != 0 else float('inf')
                })

    percentile_comparison_df = pd.DataFrame(rows)
    percentile_csv_path = os.path.join(output_dir, 'percentile_comparison.csv')
    percentile_comparison_df.to_csv(percentile_csv_path, index=False)
    print(f"Percentile comparison saved to '{percentile_csv_path}'.")

    for metric in metrics:
        df_metric = percentile_comparison_df[percentile_comparison_df['Metric'] == metric].copy()
        plt.figure(figsize=(10, 6))
        plt.plot(df_metric['Percentile'], df_metric[label1], label=f'{label1} {metric}', marker='o', color='blue')
        plt.plot(df_metric['Percentile'], df_metric[label2], label=f'{label2} {metric}', marker='o', color='orange')
        plt.xlabel('Percentiles')
        plt.ylabel('Milliseconds')
        plt.title(f'{metric} Percentile-wise Comparison')
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plot_path = os.path.join(output_dir, f'{label1}_vs_{label2}_{metric}_percentiles.png')
        plt.savefig(plot_path)
        plt.close()

    # Plot full-cycle smoothed intraday plots
    for metric in metrics:
        plt.figure(figsize=(12, 6))

        df1_metric = df1[df1['Metric'] == metric].copy()
        df2_metric = df2[df2['Metric'] == metric].copy()

        if df1_metric.empty or df2_metric.empty:
            print(f"Skipping {metric} as no data available for one machine")
            continue

        df1_metric['Smoothed'] = df1_metric['Average'].rolling(window=5, min_periods=1).mean()
        df2_metric['Smoothed'] = df2_metric['Average'].rolling(window=5, min_periods=1).mean()

        plt.plot(df1_metric['Timestamp'], df1_metric['Smoothed'], label=f'{label1} {metric}', color='blue')
        plt.plot(df2_metric['Timestamp'], df2_metric['Smoothed'], label=f'{label2} {metric}', color='orange')

        plt.gca().xaxis.set_major_formatter(plt.matplotlib.dates.DateFormatter('%H:%M'))
        plt.gcf().autofmt_xdate()

        plt.xlabel('Time')
        plt.ylabel('Smoothed Avg Delay (ms)')
        plt.title(f'{metric} Intraday Smoothed Comparison (Full Cycle)')
        plt.legend()
        plt.grid(True)
        plt.tight_layout()

        plot_path = os.path.join(output_dir, f'{label1}_vs_{label2}_{metric}_intraday_smoothed_full.png')
        plt.savefig(plot_path)
        plt.close()

def main():
    parser = argparse.ArgumentParser(description="Compare latency data between two machines")
    parser.add_argument('file1', help="Path to the first machine's latency file")
    parser.add_argument('file2', help="Path to the second machine's latency file")
    parser.add_argument('--label1', default="Machine_A", help="Label for the first machine")
    parser.add_argument('--label2', default="Machine_B", help="Label for the second machine")
    args = parser.parse_args()

    if not os.path.isfile(args.file1) or not os.path.isfile(args.file2):
        print("Error: One or both input files do not exist.")
        return

    label1 = args.label1 if args.label1 != "Machine_A" else os.path.splitext(os.path.basename(args.file1))[0]
    label2 = args.label2 if args.label2 != "Machine_B" else os.path.splitext(os.path.basename(args.file2))[0]

    df1 = parse_delay_data(args.file1)
    df2 = parse_delay_data(args.file2)

    output_dir = os.path.dirname(args.file1)
    compare_metrics(df1, df2, label1, label2, output_dir)

if __name__ == '__main__':
    main()
