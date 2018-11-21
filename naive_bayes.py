import collections

import numpy as np
import pandas as pd

df = pd.read_csv('/home/nam/ML_181/data/restaurant_wait.csv')

decisions_count = collections.Counter(df["Wait"])
prices_count = collections.Counter(df["Price"])

prices_and_decisions_table = np.zeros(shape=((len(prices_count), len(decisions_count))), dtype=int)
prices_and_decisions_table = pd.DataFrame(prices_and_decisions_table, columns=[*decisions_count],
                                          index=[*prices_count])

prices_and_decisions_table[:] = 0

for _, each_observation in pd.DataFrame(df, columns=["Price", "Wait"]).iterrows():
    prices_and_decisions_table.at[each_observation["Price"], each_observation["Wait"]] += 1


def init_list_tables(data_frame, target_name):
    decisions_count = collections.Counter(data_frame[target_name])
    list_table_count = {}

    for each_attribute_name in data_frame:
        each_attribute = df[each_attribute_name]
        att_count = collections.Counter(each_attribute)

        table_with_decisions_table = np.zeros(shape=((len(att_count), len(decisions_count))), dtype=int)

        table_with_decisions_table = pd.DataFrame(table_with_decisions_table, columns=[*decisions_count],
                                                  index=[*att_count])
        list_table_count.update({each_attribute_name: table_with_decisions_table})

    for each_attribute_name in att_name_list:

        list_table_count[each_attribute_name][:] = 0

    return list_table_count


def update_list_tables(data_frame, att_name_list, target_name):

    list_table_count = init_list_tables(data_frame, target_name)

    for each_attribute_name in att_name_list:
        for _, each_observation in pd.DataFrame(data_frame, columns=[each_attribute_name, target_name]).iterrows():
	    v = list_table_count[each_attribute_name].at[each_observation[each_attribute_name], each_observation[target_name]]

            list_table_count[each_attribute_name].at[
                each_observation[each_attribute_name], each_observation[target_name]] = v + 1

    return list_table_count


att_name_list = ["Alt", "Bar", "Fri", "Hun", "Pat", "Price", "Rain", "Res", "Type", "Est", "Wait"]

list_table_count = update_list_tables(data_frame=df, att_name_list=att_name_list, target_name="Wait")

print(list_table_count)
