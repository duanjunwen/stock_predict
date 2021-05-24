import sys
from 抓取 import grab_stock

if __name__ == '__main__':
    print('请输入股票代码:\n')
    stock_code = sys.stdin.readline()
    # print(stock_code)
    print('请输入想预测的价格(open:开盘价,high:最高价,low:最低价,close:停盘价):\n')
    predict_labels = sys.stdin.readline()
    print(predict_labels)
    grab_stock(stock_code)  # 抓取股票数据

