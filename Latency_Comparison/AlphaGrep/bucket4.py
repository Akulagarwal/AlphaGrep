import pandas as pd
import numpy as np
import sys

def bucketAnalysis(df, alpha, returnX, buckets):  # dataframe, bucketing_feature, ret_features, number_of_buckets
    df["alphaRank"] = df[alpha].rank(pct=True)
    ranges = list(np.arange(0, 1, 1.0 / buckets)) + [1.0]
    res = []
    headers = ["quantile_number", "start_point", "end_point"]
    for return_ in returnX:
        headers.append("mean_" + return_)
    for i in range(1, len(ranges)):
        startPct = ranges[i - 1]
        endPct = ranges[i]
        tempDf = df[(df["alphaRank"] >= startPct) & (df["alphaRank"] <= endPct)]
        tempList = [i, tempDf[alpha].min(), tempDf[alpha].max()]
        for return_ in returnX:
            tempList.append("%.4f" % (tempDf[return_].mean()))
        res.append(tempList)
    df = pd.DataFrame(res, columns=headers)
    return df

def parse_out_file(file_path):
    # Read file line by line to handle custom format
    data = []
    with open(file_path, 'r') as f:
        for line in f:
            # Split on " STRAT: OPTION_INFO|" to separate timestamp and fields
            parts = line.strip().split(" STRAT: OPTION_INFO|")
            if len(parts) != 2:
                print(f"Skipping malformed line: {line}")
                continue
            fields = parts[1].split("|")
            if len(fields) != 6:
                print(f"Skipping line with {len(fields)} fields: {line
    # Check if any call options were found
    if not data:
        raise ValueError("No call options found in the input file")
    
    # Create DataFrame with 6 columns
    df = pd.DataFrame(data, columns=["Symbol", "IV", "DELTA", "bid", "ask", "fut_mid"])
    
    # Convert numeric columns
    df["bid"] = pd.to_numeric(df["bid"], errors='coerce')
    df["ask"] = pd.to_numeric(df["ask"], errors='coerce')
    df["IV"] = pd.to_numeric(df["IV"], errors='coerce')
    df["DELTA"] = pd.to_numeric(df["DELTA"], errors='coerce')
    df["fut_mid"] = pd.to_numeric(df["fut_mid"], errors='coerce')
    
    # Calculate derived columns
    df["mid"] = (df["bid"] + df["ask"]) / 2
    df["spread"] = (df["ask"] - df["bid"]) / df["mid"]
    
    return df

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python bucket1.py <out_file.txt>")
        sys.exit(1)

    file_path = sys.argv[1]
    df = parse_out_file(file_path)
    print(f"Parsed DataFrame (Calls Only):\n{df}\n")  # Debug print
    result = bucketAnalysis(df, alpha="DELTA", returnX=["spread"], buckets=10)
    print(result.to_string(index=False))