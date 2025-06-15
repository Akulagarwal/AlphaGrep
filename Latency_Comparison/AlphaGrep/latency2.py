import pandas as pd
import matplotlib.pyplot as plt
import argparse
import os
from datetime import datetime

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
    df1_aligned, df2_aligned = align_time_series(df1, df2)

    if df1_aligned.empty or df2_aligned.empty:
        print("Warning: No overlapping time periods between the two datasets")
        return

    # --- Average Delay Comparison Table ---
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
    print(f"✅ Average delay comparison saved to '{avg_csv_path}'.")

    # --- Reshaped Percentile Comparison Table ---
    percentiles = ['75th%', '80th%', '85th%', '90th%', '95th%', '99th%', '999th%', '9999th%']
    reshaped_rows = []

    for metric in metrics:
        row1 = {'Label': label1, 'Metric': metric}
        row2 = {'Label': label2, 'Metric': metric}

        for perc in percentiles:
            val1 = df1_aligned[df1_aligned['Metric'] == metric][perc].mean()
            val2 = df2_aligned[df2_aligned['Metric'] == metric][perc].mean()
            if pd.isna(val1) or pd.isna(val2):
                continue
            row1[perc] = round(val1, 2)
            row2[perc] = round(val2, 2)

        reshaped_rows.append(row1)
        reshaped_rows.append(row2)

    reshaped_df = pd.DataFrame(reshaped_rows)
    reshaped_csv = os.path.join(output_dir, "percentile_comparison_clean.csv")
    reshaped_df.to_csv(reshaped_csv, index=False)
    print(f"✅ Clean reshaped percentile comparison table saved to: {reshaped_csv}")
    print(reshaped_df.to_string(index=False))

    # --- Percentile Plot ---
    for metric in metrics:
        df_metric = reshaped_df[reshaped_df['Metric'] == metric]
        if df_metric.empty:
            continue

        plt.figure(figsize=(10, 6))
        plt.plot(percentiles, df_metric[df_metric['Label'] == label1][percentiles].values.flatten(),
                 label=f'{label1} {metric}', marker='o', color='blue')
        plt.plot(percentiles, df_metric[df_metric['Label'] == label2][percentiles].values.flatten(),
                 label=f'{label2} {metric}', marker='o', color='orange')

        plt.xlabel('Percentiles')
        plt.ylabel('Milliseconds')
        plt.title(f'{metric} Percentile-wise Comparison')
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plot_path = os.path.join(output_dir, f'{label1}_vs_{label2}_{metric}_percentiles.png')
        plt.savefig(plot_path)
        plt.close()

    # --- Smoothed Intraday Plot ---
    for metric in metrics:
        plt.figure(figsize=(12, 6))

        df1_metric = df1[df1['Metric'] == metric]
        df2_metric = df2[df2['Metric'] == metric]

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

    # --- Synthetic Average Delay Percentile Analysis ---
    combined_percentile_rows = []

    for metric in metrics:
        df1_metric = df1_aligned[df1_aligned['Metric'] == metric]
        df2_metric = df2_aligned[df2_aligned['Metric'] == metric]

        merged = pd.merge(df1_metric[['Timestamp', 'Average']],
                          df2_metric[['Timestamp', 'Average']],
                          on='Timestamp',
                          suffixes=('_1', '_2'))

        if merged.empty:
            continue

        merged['AverageCombined'] = merged[['Average_1', 'Average_2']].mean(axis=1)

        percentiles_values = {}
        for perc in [75, 80, 85, 90, 95, 99, 99.9, 99.99]:
            percentile = round(perc, 4)
            val = round(merged['AverageCombined'].quantile(perc / 100.0), 2)
            label = f'{int(perc)}th%' if perc.is_integer() else f'{perc}th%'
            percentiles_values[label] = val

        row = {'Label': 'AVERAGE', 'Metric': metric}
        row.update(percentiles_values)
        combined_percentile_rows.append(row)

    if combined_percentile_rows:
        avg_combined_df = pd.DataFrame(combined_percentile_rows)
        avg_combined_path = os.path.join(output_dir, "average_combined_percentile_table.csv")
        avg_combined_df.to_csv(avg_combined_path, index=False)
        print(f"✅ Synthetic average delay percentile table saved to: {avg_combined_path}")
        print(avg_combined_df.to_string(index=False))

    # --- New Detailed Average Delay Percentile Table ---
    all_percentiles = [10, 20, 30, 40, 50, 60, 70, 80, 90, 95, 99]
    detailed_rows = []

    for label, df in [(label1, df1_aligned), (label2, df2_aligned)]:
        for metric in metrics:
            df_metric = df[df['Metric'] == metric]
            if df_metric.empty:
                continue
            row = {'Label': label, 'Metric': metric}
            for p in all_percentiles:
                perc_val = df_metric['Average'].quantile(p / 100.0)
                row[f'{p}th%'] = round(perc_val, 2)
            detailed_rows.append(row)

    detailed_df = pd.DataFrame(detailed_rows)
    detailed_percentile_csv = os.path.join(output_dir, "detailed_average_delay_percentiles.csv")
    detailed_df.to_csv(detailed_percentile_csv, index=False)
    print(f"✅ Detailed average delay percentile table saved to: {detailed_percentile_csv}")
    print(detailed_df.to_string(index=False))

def main():
    parser = argparse.ArgumentParser(description="Compare latency data between two machines")
    parser.add_argument('file1', help="Path to the first machine's latency file")
    parser.add_argument('file2', help="Path to the second machine's latency file")
    parser.add_argument('--label1', default=None, help="Label for the first machine")
    parser.add_argument('--label2', default=None, help="Label for the second machine")
    args = parser.parse_args()

    def extract_label(path):
        return os.path.basename(path).rsplit('.', 1)[0]

    if not os.path.isfile(args.file1) or not os.path.isfile(args.file2):
        print("Error: One or both input files do not exist.")
        return

    label1 = args.label1 or extract_label(args.file1)
    label2 = args.label2 or extract_label(args.file2)

    df1 = parse_delay_data(args.file1)
    df2 = parse_delay_data(args.file2)

    output_dir = os.path.dirname(args.file1) or '.'
    compare_metrics(df1, df2, label1, label2, output_dir)

if __name__ == '__main__':
    main()
