import akshare as ak

stock_zh_a_cdr_daily_df = ak.stock_zh_a_daily(symbol='sh600000', start_date='20100101', end_date='20210509')
print(stock_zh_a_cdr_daily_df)
stock_zh_a_cdr_daily_df.to_csv("sh600000.csv", sep=',', encoding='utf-8')
