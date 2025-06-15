import pandas as pd
import numpy as np
import sys
import os
import glob

def bucketAnalysis(df, alpha, returnX, buckets):
    if df.empty:
        raise ValueError("Input DataFrame is empty. Cannot perform bucket analysis.")
    if alpha not in df.columns:
        raise ValueError(f"Alpha column '{alpha}' not found in DataFrame.")
    for return_ in returnX:
        if return_ not in df.columns:
            raise ValueError(f"Return column '{return_}' not found in DataFrame.")

    df["alphaRank"] = df[alpha].rank(pct=True)
    ranges = list(np.arange(0, 1, 1.0 / buckets)) + [1.0]
    res = []
    headers = ["Quantile Number", "Start Point", "End Point"]
    for return_ in returnX:
        headers.append(f"Mean {return_} (%)")
    
    counts = []
    for i in range(1, len(ranges)):
        startPct = ranges[i - 1]
        endPct = ranges[i]
        tempDf = df[(df["alphaRank"] >= startPct) & (df["alphaRank"] <= endPct)]
        tempList = [i, f"{tempDf[alpha].min():.4f}" if not tempDf.empty else "N/A", 
                    f"{tempDf[alpha].max():.4f}" if not tempDf.empty else "N/A"]
        counts.append(len(tempDf))
        for return_ in returnX:
            mean_value = tempDf[return_].mean() * 100 if not tempDf.empty else np.nan
            tempList.append(f"{mean_value:.4f}" if not np.isnan(mean_value) else "N/A")
        res.append(tempList)
    
    result_df = pd.DataFrame(res, columns=headers)
    return result_df, counts

def parse_out_file(file_path):
    data = []
    with open(file_path, 'r') as f:
        for line in f:
            parts = line.strip().split('|')
            if len(parts) != 7:
                print(f"Skipping malformed line in {file_path}: {line.strip()}")
                continue
            timestamp, symbol, iv, delta, bid, ask, fut_mid = parts
            data.append([symbol, iv, delta, bid, ask, fut_mid])
    
    if not data:
        raise ValueError(f"No valid data parsed from {file_path}")
    
    df = pd.DataFrame(data, columns=["Symbol", "IV", "DELTA", "bid", "ask", "fut_mid"])
    numeric_cols = ["IV", "DELTA", "bid", "ask", "fut_mid"]
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    
    # Normalize IV (convert percentage to decimal if > 1)
    df["IV"] = df["IV"].apply(lambda x: x / 100 if pd.notna(x) and x > 1 else x)
    
    df["mid"] = (df["bid"] + df["ask"]) / 2
    df["spread"] = (df["ask"] - df["bid"]) / df["mid"]
    
    print(f"Parsed {file_path}: {len(df)} rows, IV range: {df['IV'].min():.4f}–{df['IV'].max():.4f}")
    return df

def bucket_analysis_iv_only(df, iv_column="IV", returnX=["spread"], buckets=10, delta_range=(0.35, 0.5)):
    min_delta, max_delta = delta_range
    filtered_df = df[(df["DELTA"].abs() >= min_delta) & (df["DELTA"].abs() <= max_delta) & df["DELTA"].notna()]
    
    row_count = len(filtered_df)
    if filtered_df.empty:
        raise ValueError(f"No data in DELTA range {min_delta} to {max_delta}")
    
    print(f"Data points in DELTA range {min_delta} to {max_delta}: {row_count}")
    
    result_df, counts = bucketAnalysis(filtered_df, alpha=iv_column, returnX=returnX, buckets=buckets)
    
    summary = pd.DataFrame([
        ["DELTA Range", min_delta, max_delta] + [""] * (len(result_df.columns) - 3),
        ["Data Points", row_count, ""] + [""] * (len(result_df.columns) - 3)
    ], columns=result_df.columns)
    
    final_df = pd.concat([summary, result_df], ignore_index=True)
    return final_df, counts

def combine_bucket_analysis(folder_path, delta_range=(0.35, 0.5), buckets=10, returnX=["spread"], iv_column="IV"):
    file_pattern = os.path.join(folder_path, "out_*_SPREAD.csv")
    files = glob.glob(file_pattern)
    if not files:
        raise ValueError(f"No files matching 'out_*_SPREAD' found in {folder_path}")
    
    all_results = []
    all_counts = []
    all_ivs = []
    total_data_points = 0
    valid_files = 0
    
    for file_path in files:
        print(f"Processing {file_path}...")
        try:
            df = parse_out_file(file_path)
            result_df, counts = bucket_analysis_iv_only(df, iv_column=iv_column, returnX=returnX, 
                                                      buckets=buckets, delta_range=delta_range)
            result_df = result_df[result_df["Quantile Number"].apply(lambda x: isinstance(x, (int, float)))]
            if len(result_df) != buckets:
                print(f"Warning: {file_path} has {len(result_df)} quantiles, expected {buckets}")
                continue
            all_results.append(result_df)
            all_counts.append(counts)
            # Collect IV values for combined ranking
            filtered_df = df[(df["DELTA"].abs() >= delta_range[0]) & (df["DELTA"].abs() <= delta_range[1]) & df["DELTA"].notna()]
            all_ivs.append(filtered_df["IV"].dropna())
            total_data_points += sum(counts)
            valid_files += 1
        except ValueError as e:
            print(f"Skipping {file_path}: {e}")
            continue
    
    if valid_files < 2:
        raise ValueError(f"Need at least 2 valid files; only {valid_files} succeeded")
    
    # Combine all IVs for global quantile boundaries
    all_ivs_combined = pd.concat([pd.Series(iv) for iv in all_ivs], ignore_index=True)
    if all_ivs_combined.empty:
        raise ValueError("No valid IV data after combining files")
    
    # Compute global IV quantiles
    quantile_boundaries = np.percentile(all_ivs_combined, np.linspace(0, 100, buckets + 1))
    print(f"Global IV quantile boundaries: {quantile_boundaries}")
    
    # Initialize combined DataFrame
    combined_df = pd.DataFrame({
        "Quantile Number": range(1, buckets + 1),
        "Start Point": ["N/A"] * buckets,
        "End Point": ["N/A"] * buckets,
        f"Mean {returnX[0]} (%)": ["N/A"] * buckets
    })
    
    # Aggregate results
    weighted_sums = np.zeros((buckets, len(returnX)), dtype=np.float64)
    total_weights = np.zeros(buckets, dtype=np.float64)
    file_contribs = np.zeros(buckets, dtype=int)
    
    for idx, (result_df, counts) in enumerate(zip(all_results, all_counts)):
        print(f"File {idx + 1}: Quantiles {result_df['Quantile Number'].tolist()}, Counts {counts}")
        for i in range(buckets):
            if counts[i] > 0:
                file_contribs[i] += 1
            for j, return_ in enumerate(returnX):
                value = result_df.iloc[i][f"Mean {return_} (%)"]
                if value != "N/A":
                    weighted_sums[i, j] += float(value) * counts[i]
            total_weights[i] += counts[i]
    
    # Warn if few files contribute
    for i in range(buckets):
        if file_contribs[i] < valid_files * 0.1:
            print(f"Warning: Quantile {i+1} has data from only {file_contribs[i]}/{valid_files} files")
    
    # Assign Start and End Points based on global IV quantiles
    for i in range(buckets):
        combined_df.at[i, "Start Point"] = f"{quantile_boundaries[i]:.4f}"
        combined_df.at[i, "End Point"] = f"{quantile_boundaries[i+1]:.4f}"
        for j, return_ in enumerate(returnX):
            if total_weights[i] > 0:
                combined_df.at[i, f"Mean {return_} (%)"] = f"{weighted_sums[i, j] / total_weights[i]:.4f}"
            else:
                combined_df.at[i, f"Mean {return_} (%)"] = "N/A"
    
    # Create summary
    summary = pd.DataFrame([
        ["DELTA Range", delta_range[0], delta_range[1]] + [""] * (len(combined_df.columns) - 3),
        ["Data Points", total_data_points, ""] + [""] * (len(combined_df.columns) - 3),
        ["Valid Files", valid_files, ""] + [""] * (len(combined_df.columns) - 3)
    ], columns=combined_df.columns)
    
    final_df = pd.concat([summary, combined_df], ignore_index=True)
    return final_df

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python buckets.py <folder_path>")
        sys.exit(1)
    
    folder_path = sys.argv[1]
    if not os.path.isdir(folder_path):
        print(f"Error: Folder '{folder_path}' does not exist.")
        sys.exit(1)
    
    try:
        delta_range = (0.35, 0.5)
        result_df = combine_bucket_analysis(folder_path, delta_range=delta_range)
        print(f"\nCombined IV Bucket Analysis (|DELTA| in {delta_range}):")
        print(result_df.to_string(index=False))
        csv_path = os.path.join(folder_path, 'combined_iv_bucket_analysis.csv')
        result_df.to_csv(csv_path, index=False)
        print(f"Saved to '{csv_path}'")
    except ValueError as e:
        print(f"Error: {e}")
        sys.exit(1)
