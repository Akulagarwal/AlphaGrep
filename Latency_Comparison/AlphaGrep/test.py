import pandas as pd
import matplotlib.pyplot as plt
import argparse
import os

def parse_delay_data(filepath):
    columns = ["Metric", "PacketCount", "Average", "Min", "Max",
               "75th%", "80th%", "85th%", "90th%", "95th%", "99th%", "999th%", "9999th%"]
    data = []

    with open(filepath, 'r') as f:
        for line in f:
            line = line.strip()
            if line.startswith(('MDDELAY', 'TOTALDELAY', 'DELAYWITHORDER')):
                parts = line.split()
                metric = parts[0]
                try:
                    row = [metric] + list(map(float, parts[1:]))
                    if len(row) == len(columns):
                        data.append(row)
                except ValueError:
                    print(f"Skipping malformed line: {line}")
    return pd.DataFrame(data, columns=columns)

def compare_metrics(df1, df2, label1, label2, output_dir):
    metrics = ['MDDELAY', 'TOTALDELAY', 'DELAYWITHORDER']

    # === AVERAGE DELAY COMPARISON (No Plot) ===
    comparison = {}
    for metric in metrics:
        avg1 = df1[df1['Metric'] == metric]['Average'].mean()
        avg2 = df2[df2['Metric'] == metric]['Average'].mean()
        comparison[metric] = {
            f'{label1}_Average': avg1,
            f'{label2}_Average': avg2,
            'Difference': avg2 - avg1,
            'Percent Change': ((avg2 - avg1) / avg1 * 100) if avg1 != 0 else float('inf')
        }

    comparison_df = pd.DataFrame(comparison).T
    print("=== Average Delay Comparison ===")
    print(comparison_df)

    avg_csv_path = os.path.join(output_dir, 'average_comparison.csv')
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
                    f'{label1}': p1,
                    f'{label2}': p2,
                    'Difference': p2 - p1,
                    'Percent Change': ((p2 - p1) / p1 * 100) if p1 != 0 else float('inf')
                })
            else:
                print(f"Warning: {p} column not found for metric {metric}")

    percentile_comparison_df = pd.DataFrame(rows)
    print("=== Percentile-wise Delay Comparison ===")
    print(percentile_comparison_df)

    percentile_csv_path = os.path.join(output_dir, 'percentile_comparison.csv')
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
        plot_path = os.path.join(output_dir, f'{label1}_vs_{label2}_{metric}_percentiles.png')
        plt.savefig(plot_path)
        plt.show()

    # === INTRADAY AVERAGE COMPARISON ===
    for metric in metrics:
        plt.figure(figsize=(10, 6))
        df1_metric = df1[df1['Metric'] == metric]
        df2_metric = df2[df2['Metric'] == metric]

        plt.plot(df1_metric.index, df1_metric['Average'], label=f'{label1} {metric}', color='blue')
        plt.plot(df2_metric.index, df2_metric['Average'], label=f'{label2} {metric}', color='orange')

        plt.xlabel('Time')
        plt.ylabel('Average Delay (ms)')
        plt.title(f'{metric} Intraday Comparison')
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plot_path = os.path.join(output_dir, f'{label1}_vs_{label2}_{metric}_intraday.png')
        plt.savefig(plot_path)
        plt.show()

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
