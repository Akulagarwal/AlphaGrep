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
    df = pd.read_csv(file_path, sep="|", header=None)
    df.columns = ["Symbol", "IV", "DELTA", "bid", "ask", "fut_mid"]
    df["bid"] = pd.to_numeric(df["bid"], errors='coerce')
    df["ask"] = pd.to_numeric(df["ask"], errors='coerce')
    df["mid"] = (df["bid"] + df["ask"]) / 2
    df["spread"] = (df["ask"] - df["bid"]) / df["mid"]

    return df

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python bucket.py <out_file.txt>")
        sys.exit(1)

    file_path = sys.argv[1]
    df = parse_out_file(file_path)
    result = bucketAnalysis(df, alpha="DELTA", returnX=["spread"], buckets=5)
    print(result.to_string(index=False))
