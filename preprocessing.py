import os
import sys
from datetime import datetime, timedelta

import pygrib
import numpy as np
import pandas as pd
from scipy.interpolate import interp1d, griddata

# 路径
meteFile = "/3clear/share/data/gfs_dataset/{begTime}/gfs.t{begH}z.pgrb2.0p25.f{hour:0>3}"
# [bedrock@node1 xair]$ tree /3clear/share/data/gfs_dataset/ -d
# /3clear/share/data/gfs_dataset/
# ├── 2024110112
# ├── 2024110212
# [bedrock@node1 xair]$ ls /3clear/share/data/gfs_dataset/2024121412 | head -2
# gfs.t12z.pgrb2.0p25.f004
# gfs.t12z.pgrb2.0p25.f005

enviFile = "/archive/share/Data/base/senvi/%Y/%Y%m/obs_envi_%Y%m%d%H.txt"
# [bedrock@node1 xair]$ head -n 1 /archive/share/Data/base/senvi/2025/202501/obs_envi_2025010100.txt 
# 110000244,30.00,66.00,0.40,63.00,3.00,9.00,22.00

# 站点文件
siteFile = "./dataset/site_all.txt"
# [bedrock@node1 xair]$ head -n 2 dataset/site_all.txt 
# code           lon     lat   city
# 110000041 116.2202 40.2915 北京市

meteMeta = {
    "PBLH":dict(name="Planetary boundary layer height", typeOfLevel="surface", level=[0]),
    "RAINNCV":dict(name="Total Precipitation", typeOfLevel="surface", level=[0]),
    "SWDOWN":dict(name="Downward short-wave radiation flux", typeOfLevel="surface", level=[0]),
    "SLP":dict(name="Pressure reduced to MSL", typeOfLevel="meanSea", level=[0]), # 数据错误
    "U10":dict(name="10 metre U wind component", typeOfLevel="heightAboveGround", level=[10]),
    "V10":dict(name="10 metre V wind component", typeOfLevel="heightAboveGround", level=[10]),
    "T2":dict(name="2 metre temperature", typeOfLevel="heightAboveGround", level=[2]),
    "RH2":dict(name="2 metre relative humidity", typeOfLevel="heightAboveGround", level=[2]),
}

meteVars = ["PBLH", "RAINNCV", "SWDOWN", "SLP", "U10", "V10", "T2", "RH2"]

def read_input_data(begTime, min_T, lead_hour):
    '''读取 历史观测数据 + 预报气象数据'''
    C_poll, C_mete = 6, 8

    obsData = []
    sTime = begTime - timedelta(hours=min_T)
    for i in range(min_T): # 读取历史观测数据, 可以多读取一些？
        thisTime = sTime + timedelta(hours=i)
        iData = pd.read_csv(thisTime.strftime(enviFile))
        obsData.append(iData.iloc[:, 1:7].values)
    site_codes = iData.iloc[:, 0].values

    site_info = pd.read_csv(siteFile, delim_whitespace=True)
    site_info = site_info[site_info['code'].isin(site_codes)]

    # PM25,PM10,CO,NO2,SO2,O3,PBLH,RAINNCV,SWDOWN,SLP,U10,V10,T2,RH2
    data = np.ones((min_T+lead_hour, len(site_codes), C_poll + C_mete)) * -999.  # 初始化数据

    data[:min_T, :, :C_poll] = np.array(obsData)  # 历史观测数据

    stations = {
        "codes": site_info["code"].values,
        "lons": site_info["lon"].values,
        "lats": site_info["lat"].values
    }

    # 不同时效的GFS数据拼接
    gfs_files, xtime = [], []
    utc_ime = begTime - timedelta(hours=8)  # UTC时间
    gfs_time = utc_ime  - timedelta(hours=utc_ime.hour % 6)  # GFS起报时次
    gfs_times = [gfs_time - timedelta(hours=i*6) for i in range(5)]
    for i in range(min_T+lead_hour): # 读取历史观测数据, 可以多读取一些？
        thisTime = sTime + timedelta(hours=i)
        xtime.append(thisTime)
        utc_ime = thisTime - timedelta(hours=8)  # UTC时间
        has_gfs = False
        for gfs_time in gfs_times: # 找到最新的GFS数据
            hour = int((utc_ime - gfs_time).total_seconds() / 3600)
            gfs_file = utc_ime.strftime(meteFile).format(begTime=gfs_time.strftime("%Y%m%d%H"), begH=gfs_time.hour, hour=hour)
            if os.path.exists(gfs_file):
                has_gfs = True
                gfs_files.append(gfs_file)
                break
        if not has_gfs:
            gfs_files.append("---")
   
    # 预报气象数据
    gfsSiteData = read_all_time_gfs_data(gfs_files, stations, meteVars)
    data[..., C_poll:C_poll+C_mete] = gfsSiteData

    return xtime, site_codes, site_info["city"].values, data

def read_all_time_gfs_data(gfs_files, stations, meteVars):
    ''' 获取所有站点一个起报时次的所有预报数据 '''
    # read gfs variables # time, site, vars
    meteDataSet, xlat, xlon = read_raw_gfs(gfs_files, meteVars)
    # Interpolates grib data to site
    gfsSiteData = interpolate_grid_to_site(meteDataSet, xlat, xlon, stations['lats'], stations['lons'], meteVars)
    gfsSiteData = fill_gfs_missing_data(gfsSiteData, meteVars)
    oData = np.array([gfsSiteData[ikey] for ikey in meteVars])
    return np.transpose(oData, (2, 1, 0)) # [var, site, time] => [time, site, var]

def read_raw_gfs(gfs_files, meteVars, region={"lon1":70, "lon2":140, "lat1":10, "lat2":60}):
    gfsDataSet, xlat, xlon = {}, None, None
    for thisFile in gfs_files:
        if os.path.exists(thisFile):
           gfsdata = pygrib.open(thisFile)
           for ivar in meteVars:
               try:
                   gribData = gfsdata.select(**meteMeta[ivar])[0]
                   data, xlat, xlon = gribData.data(**region)
                   gfsDataSet.setdefault(ivar, []).append(data)
               except ValueError:
                   print(ivar + " is not existence in " + thisFile) 
                   gfsDataSet.setdefault(ivar, []).append(None)
        else:
            for ivar in meteVars:
                gfsDataSet.setdefault(ivar, []).append(None)
    return gfsDataSet, xlat, xlon # var, time, lat, lon

def interpolate_grid_to_site(gfsDataSet, xlat, xlon, lats, lons, meteVars):
    gfsSiteData = {} # var, site, time
    for ikey in meteVars: # 变量
        for gfsData in gfsDataSet[ikey]: # 不同的时间的格点数据
            if xlon is not None:
                points = np.stack((xlon.flatten(), xlat.flatten()), axis=1)
            iData = np.ones(len(lats))*np.nan # 站点数据
            if gfsData is not None:
                #iData = griddata(points, gfsData.flatten(), (lons, lats), method='linear')
                iData = griddata(points, gfsData.flatten(), (lons, lats), method='nearest')
            gfsSiteData.setdefault(ikey, []).append(iData)
        gfsSiteData[ikey] = np.stack(gfsSiteData[ikey], axis=-1) # 增加时间维度
    return gfsSiteData # [var, site, time]

def fill_gfs_missing_data(gfsSiteData, meteVars):
    for ikey in meteVars: # 变量
        for i in  range(gfsSiteData[ikey].shape[0]): # 站点
            serial = gfsSiteData[ikey][i,:]
            valid = ~np.isnan(serial)
            num = len(serial[valid])
            if  num != len(serial) and num > len(serial)//3:
                x = np.linspace(0, len(serial)-1, num=len(serial))
                f = interp1d(x[valid], serial[valid], kind='cubic')
                if valid[-1] and valid[0]:
                   serial = f(x)
                else:
                   bound = np.where(valid)[0][-1]+1
                   serial[:bound] = f(x[:bound])
            serial[np.isnan(serial)] = -999.
            gfsSiteData[ikey][i,:] = serial
    return gfsSiteData # [var, site, time]

def export_data_to_csv(xtime, site_codes, city_names, data, meteVars, filename="output.csv"):
    time_len, site_len, _ = data.shape
    records = []

    for t_idx in range(time_len):
        for s_idx in range(site_len):
            row = [xtime[t_idx].strftime("%Y%m%d%H"), site_codes[s_idx], city_names[s_idx]]
            row.extend(data[t_idx, s_idx, :])
            records.append(row)

    columns = ["time", "code", "city"] + meteVars
    df = pd.DataFrame(records, columns=columns)
    df.to_csv(filename, index=False, float_format="%.2f")

def read_all_data(fileName, mean):
    '''读取 历史观测数据 + 预报气象数据，并计算城市平均，添加时间周期特征，返回 numpy'''
    C_out = 6
    siteData = pd.read_csv(fileName)

    # 将 -999 替换为 np.nan
    siteData.replace(-999., np.nan, inplace=True)

    # 计算城市均值
    cityData = siteData.groupby(['time', 'city']).mean().reset_index()
        
    # 按照 city 顺序排序
    city_info = pd.read_csv("./dataset/city.csv")
    city_order = city_info['cityName'].tolist()

    cityData = cityData[cityData['city'].isin(city_order)]
    cityData['city'] = pd.Categorical(cityData['city'], categories=city_order, ordered=True)
    cityData = cityData.sort_values(['time', 'city'])

    # 解析时间
    cityData['datetime'] = pd.to_datetime(cityData['time'], format="%Y%m%d%H")

    # 添加时间周期特征
    cityData['hour_cos'] = np.cos(2 * np.pi * cityData['datetime'].dt.hour / 24)
    cityData['month_cos'] = np.cos(2 * np.pi * cityData['datetime'].dt.month / 12)

    # 删除不需要的列
    drop_cols = ['time', 'city', 'datetime', 'code']
    final_data = cityData.drop(columns=drop_cols)
    final_data = np.nan_to_num(final_data, nan=-999.)
    oData = final_data.reshape(-1, city_info.shape[0], final_data.shape[-1])
    

    # 处理前6个特征的 log（排除缺失）
    valid_mask = oData[..., :6] != -999
    oData[..., :6][valid_mask] = np.log(oData[..., :6][valid_mask] + 1e-6)

    # 用 mean 替换缺失值，仅前16个特征
    missing_mask = (oData == -999)
    oData[missing_mask[..., :16]] = np.broadcast_to(mean[..., :16], oData.shape)[missing_mask[..., :16]]
    return oData

LEAD_HOUR = 48
site_fearture_file = "input/site_%Y%m%d%H.csv"
if __name__ == "__main__":
    begTime = datetime.strptime(sys.argv[1], "%Y%m%d%H")
    # begTime = pd.to_datetime("2025-01-01 00:00")
    min_T = 8
    lead_hour = LEAD_HOUR
    if os.path.exists(begTime.strftime(site_fearture_file)): exit()
    xtime, site_codes, city_names, data = read_input_data(begTime, min_T, lead_hour)
    export_data_to_csv(xtime, site_codes, city_names, data, 
                       meteVars=["PM2.5", "PM10", "CO", "NO2", "SO2", "O3"] + meteVars,
                       filename=begTime.strftime(site_fearture_file))
