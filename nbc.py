import pandas as pd

from src_NBC.naive_bayes import *


df = pd.read_csv('/home/nam/ML_181/data/restaurant_wait.csv')


att_name_list = df.columns.tolist()

# list_table_count = update_list_tables(data_frame=df, att_name_list=att_name_list, target_name="Wait")


list_table_count = update_list_tables(data_frame=df, target_name="Wait")

print(list_table_count)