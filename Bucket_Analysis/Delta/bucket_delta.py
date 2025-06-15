import pandas as pd
import numpy as np
import sys
import os
import re  # Importing regex module

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
    
    for i in range(1, len(ranges)):
        startPct = ranges[i - 1]
        endPct = ranges[i]
        tempDf = df[(df["alphaRank"] >= startPct) & (df["alphaRank"] <= endPct)]
        tempList = [i, f"{tempDf[alpha].min():.4f}", f"{tempDf[alpha].max():.4f}"]
        for return_ in returnX:
            mean_value = tempDf[return_].mean() * 100 if not tempDf.empty else np.nan
            tempList.append(f"{mean_value:.4f}" if not np.isnan(mean_value) else "N/A")
        res.append(tempList)
    
    result_df = pd.DataFrame(res, columns=headers)
    return result_df

def parse_out_file(file_path):
    data = []
    with open(file_path, 'r') as f:
        for line in f:
            parts = line.strip().split('|')
            if len(parts) != 7:
                #print(f"Skipping line with {len(parts)} fields: {line}")
                continue
            # Expected format: timestamp,symbol,IV,DELTA,bid,ask,fut_mid
            timestamp, symbol, iv, delta, bid, ask, fut_mid = parts
            data.append([symbol, iv, delta, bid, ask, fut_mid])
    
    if not data:
        raise ValueError("No valid data parsed from the input file.")
    
    df = pd.DataFrame(data, columns=["Symbol", "IV", "DELTA", "bid", "ask", "fut_mid"])
    # Convert to numeric, handling errors
    numeric_cols = ["IV", "DELTA", "bid", "ask", "fut_mid"]
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    
    # Compute derived columns
    df["mid"] = (df["bid"] + df["ask"]) / 2
    df["spread"] = (df["ask"] - df["bid"]) / df["mid"]
    
    return df

def filter_by_option_type(df):
    # Using regex to match symbols that end with 'C' for Calls and 'P' for Puts, allowing digits before the 'C' or 'P'
    calls_df = df[df["Symbol"].str.match(r'.*C\d+$')]  # Matches symbols ending with 'C' followed by digits
    puts_df = df[df["Symbol"].str.match(r'.*P\d+$')]   # Matches symbols ending with 'P' followed by digits
    
    # Debug output to check how many calls and puts are found
    print(f"Total options: {len(df)}")
    print(f"Calls: {len(calls_df)}, Puts: {len(puts_df)}")
    
    return calls_df, puts_df

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python script.py <input_file>")
        sys.exit(1)
    
    file_path = sys.argv[1]
    if not os.path.isfile(file_path):
        print(f"Error: File '{file_path}' does not exist.")
        sys.exit(1)
    
    try:
        df = parse_out_file(file_path)
        
        # Debug output to check the DataFrame
        #print(f"DataFrame after parsing (rows): {len(df)}")
        #print(df.head())  # Show first few rows
        
        # Filter calls and puts
        calls_df, puts_df = filter_by_option_type(df)
        
        # Check if filtered data is empty
        if calls_df.empty:
            print("No calls found in the data.")
        if puts_df.empty:
            print("No puts found in the data.")
        
        # Perform bucket analysis for calls and puts if they are not empty
        if not calls_df.empty:
            calls_result = bucketAnalysis(calls_df, alpha="DELTA", returnX=["spread"], buckets=10)
            print("Bucket Analysis for Calls:")
            print(calls_result.to_string(index=False))
            calls_csv_path = os.path.join(os.path.dirname(file_path) or '.', 'calls_bucket_analysis.csv')
            calls_result.to_csv(calls_csv_path, index=False)
            print(f"Bucket analysis for Calls saved to '{calls_csv_path}'.")
        
        if not puts_df.empty:
            puts_result = bucketAnalysis(puts_df, alpha="DELTA", returnX=["spread"], buckets=10)
            print("\nBucket Analysis for Puts:")
            print(puts_result.to_string(index=False))
            puts_csv_path = os.path.join(os.path.dirname(file_path) or '.', 'puts_bucket_analysis.csv')
            puts_result.to_csv(puts_csv_path, index=False)
            print(f"Bucket analysis for Puts saved to '{puts_csv_path}'.")
        
    except ValueError as e:
        print(f"Error: {e}")
        sys.exit(1)
