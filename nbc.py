from src_NBC.naive_bayes import *

df = pd.read_csv('/home/nam/ML_181/data/restaurant_wait.csv')

att_target = 'Wait'
att_name_list = df.columns.tolist()
att_name_list.remove(att_target)

decisions_count, list_table_count = update_list_tables(data_frame=df, target_name=att_target)

result = {}

# X = df.iloc[11:]
X11 = df.iloc[11:].to_dict('records')[0]
for each_class in decisions_count.keys():
    s = 0
    for each_attribute_name in att_name_list:
        s += np.log(list_table_count[each_attribute_name].at[\
                        X11[each_attribute_name], each_class])

        s -= np.log(np.sum(list_table_count[each_attribute_name][each_class].tolist()))

    result[each_class] = np.log(decisions_count[each_class]) + s

print(max(result))
