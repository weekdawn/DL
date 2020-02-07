# *-* coding:utf-8 *-*
import pandas as pd
import numpy as np
import keras.preprocessing.text as T
import pickle


def data_preprocess(file):
    one_three_month_data = pd.read_excel(file, sheet_name=0)
    three_five_month_data = pd.read_excel(file, sheet_name=1)
    mark_data = pd.read_excel(file, sheet_name=2, usecols=[0,1,2,5,7])
    origin_data = pd.concat([one_three_month_data,three_five_month_data],axis=0,ignore_index=True)

    # 全部转化为字符串
    origin_data.astype(str)
    mark_data.astype(str)

    # 全部小写
    origin_data['kpi_id'].str.lower()
    mark_data['kpi_id'].str.lower()

    # 修改数据集中的列顺序使一致
    order = ["unit_id", "kpi_id", "db_time", "event_title", "remark"]
    origin_data = origin_data[order]
    mark_data = mark_data[order]

    # 去掉重复行和空行
    origin_data = origin_data.drop_duplicates(subset=['unit_id', 'kpi_id', 'event_title'], keep='first')
    origin_data = origin_data.dropna(axis=0, how='all')
    mark_data = mark_data.drop_duplicates(subset=['unit_id', 'kpi_id', 'event_title'], keep='first')
    mark_data = mark_data.dropna(axis=0, how='all')

    mark_data["id"] = mark_data["unit_id"] + mark_data["kpi_id"]
    origin_data["id"] = origin_data["unit_id"] + origin_data["kpi_id"]
    #文本转数字
    origin_data["id2num"] = origin_data["id"].apply(lambda x: T.one_hot(x, 10))
    mark_data["id2num"] = mark_data["id"].apply(lambda x: T.one_hot(x, 10))
    # 统一时间格式
    origin_data.db_time = origin_data.db_time.apply(lambda x: x[:10])
    # 根据日期切分原始告警id
    gp = origin_data["id"].groupby(mark_data["db_time"])

    # 按告警时间切片
    index = 0
    mark_dic = {}

    for i in gp.count().index:
        temp = origin_data[(origin_data["db_time"] == i)]
        mark_dic[i] = temp["id"].values
        index += 1

    # 告警数据发生的前一天与后一天的数据，组成告警事务
    mark_dic_three = {}
    mark_dic_three[list(mark_dic.keys())[0]] = list(mark_dic.values())[:2]
    for i in range(1, len(mark_dic) - 1):
        mark_dic_three[list(mark_dic.keys())[i]] = [list(mark_dic.values())[j] for j in range(i - 1, i + 2)]
    mark_dic_three[list(mark_dic.keys())[-1]] = list(mark_dic.values())[-2:]

    # 把临近3天的数据合到一个列表
    for key, values in mark_dic_three.items():
        new = []
        for i in range(len(values)):
            new.extend(list(mark_dic_three[key])[i])
        mark_dic_three[key] = new

    #构造数据集
    x_set = []
    y_set = []
    mard_ids = mark_data.groupby(mark_data["id"]).count().index
    for mid in mard_ids:
        #每个标注id映射的原始告警日期
        mdates = mark_data[(mark_data["id"] == mid)]["db_time"].unique()
        #取每个标注id出现的所有日期的告警事务，记标签
        for md in range(len(mdates)):
            x_set.append(pd.DataFrame(mark_dic_three[mdates[0]]).values)
            y_set.append(mid)
    x_train = x_set[:8000]
    x_test = x_set[8001:]
    y_train = y_set[:8000]
    y_test = y_set[8001:]

    #保存数据
    with open("./train_test_set.pkl", "wb") as f:
        pickle.dump((x_train, x_test, y_train, y_test), f)

    return x_train, x_test, y_train, y_test
