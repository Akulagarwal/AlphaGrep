
import pandas as pd

def parse_log_line(line):
    parts = line.split("OPTION_INFO|")[1]
    columns = parts.strip().split("|")  
    return columns

def bucketAnalysis(df, alpha, returnX, buckets):#dataframe,bucketing_feature,ret_features,number_of_buckets
        df["alphaRank"] = df[alpha].rank(pct = True)
        ranges = list(np.arange(0,1,1.0/buckets)) + [1.0]
        res = []
        headers = ["quantile_number", "start_point", "end_point"]
        for return_ in returnX:
            headers.append("mean_" + return_)
        for i in range(1, len(ranges)):
            startPct = ranges[i-1]
            endPct = ranges[i]
            tempDf = df[(df["alphaRank"] >= startPct) & (df["alphaRank"] <= endPct)]
            tempList =  [i, tempDf[alpha].min(), tempDf[alpha].max()]
            for return_ in returnX:
                tempList.append("%.1f"%(tempDf[return_].mean()))
    
            res.append(tempList)
        df = pd.DataFrame( res, columns = headers)
        return df
def process_log_file(file_path):
    data = []
    with open(file_path, "r") as file:
        for line in file:
            columns = parse_log_line(line)
            data.append(columns)
    df = pd.DataFrame(data, columns=["Symbol", "IV", "DELTA", "bid", "ask", "fut_mid"])
    return df

