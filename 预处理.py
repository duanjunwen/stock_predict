import csv
import pandas as pd
from sklearn.model_selection import train_test_split


def read_csv_to_dataframe(filename):
    tmp_lst = []
    with open(filename, 'r') as f:
        reader = csv.reader(f)
        for row in reader:
            tmp_lst.append(row)
        f.close()
    stock_df = pd.DataFrame(tmp_lst[1:], columns=tmp_lst[0])
    return stock_df


def build_labels(filename):
    stock_df = read_csv_to_dataframe(filename)

    # print(stock_df)
    increase_or_decrease_lable = []  # 以开盘价为标准
    average_price_labels = []

    # Decrease 为 0. increase 为 1
    for index, row in stock_df.iterrows():
        # print(index, row)
        if index == 0:
            increase_or_decrease_lable.append([row['date'], 0])
        else:
            # print(index, stock_df.iloc[index - 1])
            if row['open'] <= stock_df.iloc[index - 1]['open']:
                increase_or_decrease_lable.append([row['date'], 0])
            if row['open'] > stock_df.iloc[index - 1]['open']:
                increase_or_decrease_lable.append([row['date'], 1])
        average_price_labels.append([row['date'], (float(row['open']) + float(row['close'])) / 2])
    increase_or_decrease_lable_df = pd.DataFrame(increase_or_decrease_lable, columns=['date', 'State'])
    average_price_labels_df = pd.DataFrame(average_price_labels, columns=['date', 'Avg_price'])

    xtrain, xtest, ytrain_increase_decrease, ytest_increase_decrease = train_test_split(stock_df,
                                                                                        increase_or_decrease_lable_df,
                                                                                        test_size=0.2, random_state=1,
                                                                                        shuffle=False)
    xtrain.to_csv("sh600000_x_train.csv", sep=',', encoding='utf-8')
    xtest.to_csv("sh600000_x_test.csv", sep=',', encoding='utf-8')
    ytrain_increase_decrease.to_csv("sh600000_y_train_increase_decrease.csv", sep=',', encoding='utf-8')
    ytest_increase_decrease.to_csv("sh600000_y_test_increase_decrease.csv", sep=',', encoding='utf-8')
    # average_price_labels_df.to_csv("sh600000_avg_price_label.csv", sep=',', encoding='utf-8')

    xtrain, xtest, ytrain_avg_price, ytest_avg_price = train_test_split(stock_df, average_price_labels_df,
                                                                        test_size=0.2,
                                                                        random_state=1,
                                                                        shuffle=False)
    ytrain_avg_price.to_csv("sh600000_y_train_avg_price.csv", sep=',', encoding='utf-8')
    ytest_avg_price.to_csv("sh600000_y_test_avg_price.csv", sep=',', encoding='utf-8')


# 选取前60天的作为feature
def pre_processing_df(old_df):
    columns = old_df.columns.values[1:]  # 旧列标
    # print(columns)
    new_df_columns = ['date']  # 生成新df的列index
    count = 60
    for name in columns:
        curr_name_list = []
        while count > 0:
            curr_name_list.append(f'{name}-{count}')
            count -= 1
        count = 60
        new_df_columns = new_df_columns + curr_name_list
    new_df_dict = {column_name: [] for column_name in new_df_columns}
    # len(old_df) + 1，还有第一行是index
    for i in range(60, len(old_df)):  # 前0-29提供信息，从第30行开始
        pre_thirty = old_df.iloc[i - 60: i]  # 取下old_df前30行
        new_df_dict['date'].append(old_df.iloc[i]['date'])  # 先加上day 从第60天开始
        # print(pre_thirty)
        for index, curr_column in pre_thirty.iteritems():  # 按列遍历
            if index == 'date':
                continue
            count = 60
            the_column = list(curr_column)
            for j in range(len(the_column)):
                new_df_dict[f'{index}-{count}'].append(the_column[j])
                count -= 1
    new_df = pd.DataFrame(new_df_dict)
    # print(new_df)
    return new_df


#  根据要求的feature和interval切割
def cut_by_feature_and_interval(original_df, feature_name_list, interval):  # , cases_interval
    interval_start = interval  # 每个weather feature开始的位置
    # case_start = cases_interval  # 每个daily_case开始的位置

    columns = feature_name_list
    weather_features = list(columns)
    new_df_columns = []  # 生成新df的列column index

    interval_count = interval_start  # weather name 的计数器
    for name in weather_features:  # 除了dailly cases的weather feature name list
        curr_name_list = []
        while interval_count > 0:
            curr_name_list.append(f'{name}-{interval_count}')
            interval_count -= 1
        interval_count = interval_start  # 重置计数器
        new_df_columns = new_df_columns + curr_name_list

    new_df_dict = {column_name: [] for column_name in new_df_columns}  # 生成511个key的字典
    
    for index, curr_columns in original_df.iteritems():  # 按列遍历
        the_column = list(curr_columns)
        if index in new_df_dict.keys():
            new_df_dict[index] = the_column
    new_df = pd.DataFrame(new_df_dict)
    # print(new_df)
    return new_df

'''''''''
file_name = 'sh600000.csv'
# build_labels(file_name)
'''''''''