import os
import glob
import pickle

import pandas as pd
import numpy as np

# 读取站点信息
def load_site_info(site_file):
    site_df = pd.read_csv(site_file, delim_whitespace=True)
    return site_df.set_index('code')

# 读取某一年某站点数据
def load_site_data(file):
    df = pd.read_csv(file)
    df['hour'] = df['dateTime'].astype(str).str[-2:].astype(int)
    df['month'] = df['dateTime'].astype(str).str[4:6].astype(int)
    
    # 周期化编码: sin/cos
    df['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 24)
    df['month_cos'] = np.cos(2 * np.pi * df['month'] / 12)

    df = df.drop(columns=['dateTime', 'hour', 'month'])
    return df

site_info = load_site_info("site.txt")

# 读取 city.csv 的城市顺序
city_df = pd.read_csv("city.csv")
city_order = city_df['cityName'].tolist()

# 存储所有数据
cityData_list = []

for year in range(2019, 2026):
    print(year)
    tmp = []
    for site_id in site_info.index:
        file_path = f"/archive/share/projs/foml/v1.1.0/history/{site_id}/{year}010100.csv"
        df = load_site_data(file_path)
        df['site'] = site_id
        df['city'] = site_info.loc[site_id, 'city']
        if year == 2019:
           df['time'] = pd.date_range(start=f'{year-1}-01-02', periods=len(df), freq='H')
        else:
           df['time'] = pd.date_range(start=f'{year-1}-01-01', periods=len(df), freq='H')
        tmp.append(df)
    # 合并成大表
    siteData = pd.concat(tmp, ignore_index=True)

    # 保存站点级别数据 siteData[Time, site, variable]
    siteData = siteData.set_index(['time', 'site'])

    # 按城市聚合
    cityData = siteData.groupby(['time', 'city']).mean().reset_index()

    # ✅ 按city.csv中的城市顺序排序
    cityData['city'] = pd.Categorical(cityData['city'], categories=city_order, ordered=True)
    cityData = cityData.sort_values(['time', 'city']).set_index(['time', 'city'])

    print(cityData.shape, cityData.shape[0]/334/24)
    cityData_list.append(cityData)

#data = pd.concat(cityData_list, ignore_index=True)
data = pd.concat(cityData_list)

# 保存
with open('city_data.pkl', 'wb') as f:
    pickle.dump(data, f)

