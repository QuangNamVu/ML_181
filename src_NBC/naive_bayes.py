import collections

import numpy as np
import pandas as pd

# df = pd.read_csv('/home/nam/ML_181/data/restaurant_wait.csv')


def init_list_tables(data_frame, target_name):
    decisions_count = collections.Counter(data_frame[target_name])
    list_table_count = {}

    att_name_list = data_frame.columns.tolist()

    for each_attribute_name in data_frame:
        each_attribute = data_frame[each_attribute_name]
        att_count = collections.Counter(each_attribute)

        table_with_decisions_table = np.zeros(shape=((len(att_count), len(decisions_count))), dtype=int)

        table_with_decisions_table = pd.DataFrame(table_with_decisions_table, columns=[*decisions_count],
                                                  index=[*att_count])
        list_table_count.update({each_attribute_name: table_with_decisions_table})

    for each_attribute_name in att_name_list:

        list_table_count[each_attribute_name][:] = 0

    return list_table_count


def update_list_tables(data_frame, target_name):

    list_table_count = init_list_tables(data_frame, target_name)
    att_name_list = data_frame.columns.tolist()

    att_name_list.remove(target_name)

    for each_attribute_name in att_name_list:

        for _, each_observation in pd.DataFrame(data_frame, columns=[each_attribute_name, target_name]).iterrows():
            list_table_count[each_attribute_name].at[
                each_observation[each_attribute_name], each_observation[target_name]] += 1

    return list_table_count
