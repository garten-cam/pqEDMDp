import os
import sqlite3
import pandas as pd

# 
def read_data_and_transfos(kiln, end_date='20220919'):
  data_DT = pd.read_csv(os.path.join(f"C:\Expert System Codes\Prism_extracts\Data",
                                     f"{kiln}_PRISM_DT_Tags_full_{end_date}.csv"))
  return data_DT


def read_df_from_sqlite(filepath, table, cols='*', condition=None):
    try:
        """Read data from SQLite """
        cnx = sqlite3.connect(filepath)
        if condition is None:
          df = pd.read_sql_query(f'SELECT {cols} FROM {table};', cnx)
        else:
          df = pd.read_sql_query(f'SELECT {cols} FROM {table} WHERE {condition};', cnx)
        cnx.close()
        return df

    except Exception as e:
        print("Problem reading " + table + " data from SQLite", '\n', e)
        return False


if __name__ == "__main__":
    kiln = 'BEAIKL1'
    col = 'timestamp'
    start_date = "'2022-08-30 00:00:00'"
    end_date = "'2022-08-30 15:00:00'"
    condition = f'{col} BETWEEN {start_date} AND {end_date}'
    include_thermocouples = False

    filepath_2s = f"C:\Expert System Codes\Prism_extracts\RawData\\{kiln}_RD_02s_20220101_20220501.db"
    filepath_2s_1 = f"C:\Expert System Codes\Prism_extracts\RawData\\{kiln}_RD_02s_20220501_20220901.db"
    filepath_60s = f"C:\Expert System Codes\Prism_extracts\RawData\\{kiln}_RD_60s_20220501_20220901.db"
    filepath_10s = f"C:\Expert System Codes\Prism_extracts\RawData\\{kiln}_RD_10s_20220501_20220901.db"

    df_transfos = read_data_and_transfos(kiln)

    col_dt = ['Solid fuel - Fuel A - NCV', 'Solid fuel - Fuel B - NCV', 'Solid fuel - NCV', 'Timestamp',
              'Combustion shaft', 'Cycle - Number']
    df_transfos = df_transfos[col_dt]
    df_transfos = df_transfos.rename(columns={"Timestamp": "timestamp"})
    print(f"Shape df_transfos: {df_transfos.shape}")

    selection = ('%Speed%', '%FA0%', '%Time%')
    df_2s = read_df_from_sqlite(filepath_2s_1, "data_2s", condition=condition)
    print(f"Shape df_2s: {df_2s.shape}")

    df_2s['Cycle - Number'] = df_2s[df_2s.columns[df_2s.columns.str.contains('_loads-counter_calc')]]
    df_2s.drop(['BE.AI.SP.KL1_CO-raw-channel_ana', 'BE.AI.SP.KL1_SO2-raw-channel_ana'], axis=1, inplace=True)
    col_2s = df_2s.columns[df_2s.columns.str.contains('(?=.*Time)|(?=.*Speed)|(?=.*FA0)|(?=.*WE1314-measured)|'
                                                      '(?=.*WE1314-setpoint)|(?=.*raw)|(?=.*PT)|(?=.*dig)|(?=.*loads)|'
                                                      '(?=.*strokes)|(?=.*timestamp)')]
    data = df_2s[col_2s].merge(df_transfos, on='timestamp', how='left')

    df_10s = read_df_from_sqlite(filepath_10s, "data_10s", condition=condition)
    print(f"Shape df_10s: {df_10s.shape}")
    col_10s = df_10s.columns[df_10s.columns.str.contains('(?=.*timestamp)|(?=.*level-rad)|(?=.*TT)|(?=.*fumes)|'
                                                         '(?=.*lime)')]
    data = data.merge(df_10s[col_10s], on='timestamp', how='left')

    df_60s = read_df_from_sqlite(filepath_60s, "data_60s", condition=condition)
    print(f"Shape df_60s: {df_60s.shape}")
    col_60s = df_60s.columns[df_60s.columns.str.contains('(?=.*timestamp)|(?=.*tt)|(?=.*T_blow)')]
    data = data.merge(df_60s[col_60s], on='timestamp', how='left')

    data = data.fillna(method='ffill').dropna()
    print(f"Shape data: {data.shape}")

    col_I = data.columns[data.columns.str.contains('(?=.*Speed)|(?=.*FA0)|(?=.*NCV)|(?=.*stone-weight)|'
                                                   '(?=.*WE1314-setpoint)|(?=.*strokes)')]
    if include_thermocouples:
        col_V = data.columns[data.columns.str.contains('(?=.*WE1314-measured)|(?=.*raw)|(?=.*PT)|(?=.*TT)|'
                                                       '(?=.*fumes)|(?=.*level-rad)|(?=.*lime)|(?=.*tt)')]
    else:
        col_V = data.columns[data.columns.str.contains('(?=.*WE1314-measured)|(?=.*raw)|(?=.*PT)|(?=.*TT)|'
                                                       '(?=.*fumes)|(?=.*level-rad)|(?=.*lime)')]

    print(col_I)
    print(len(col_I))
    print(col_V)
    print(len(col_V))
    print(col_2s)
    print(col_10s)
    print(col_60s)

