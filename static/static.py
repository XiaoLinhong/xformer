import xarray as xr
import pandas as pd
import numpy as np
import cartopy.io.shapereader as shpreader
from scipy.interpolate import griddata

def read_city_area(shp_file):
    area_dict = {}
    reader = shpreader.Reader(shp_file)
    for record in reader.records():
        city = record.attributes.get('city')
        if city:
            area_dict[city] = area_dict.get(city, 0) + record.attributes.get('Shape_Area', 0)
    return area_dict

def read_city_list(city_file):
    df = pd.read_csv(city_file, encoding='utf-8', dtype={'code': int})
    df.columns = ["code", "cityName", "lon", "lat"]
    return df

def interpolate_hgt(city_df, geo_file):
    geo_ds = xr.open_dataset(geo_file)
    HGT_M = geo_ds['HGT_M'].values[0]
    XLAT_M = geo_ds['XLAT_M'].values[0]
    XLONG_M = geo_ds['XLONG_M'].values[0]

    lons = XLONG_M.flatten()
    lats = XLAT_M.flatten()
    hgts = HGT_M.flatten()

    city_coords = city_df[['lon', 'lat']].values
    city_df["hgt"] = griddata(
        points=(lons, lats),
        values=hgts,
        xi=city_coords,
        method='linear'
    )
    return city_df

def read_landuse_data(landuse_codes, base_path):
    data = {}
    for code, name in landuse_codes.items():
        file_path = f"{base_path}/eiss_lndc_{code}.nc"
        ds = xr.open_dataset(file_path)
        data[name] = ds["cnts"].values
    return data

def main():
    # 读取城市面积
    area_dict = read_city_area("/archive/share/projs/mask/shp/city.shp")

    # 城市列表
    city_df = read_city_list("/public/home/bedrock/lava/module/ens/perb/naqp/city.csv")

    # 插值获取海拔
    city_df = interpolate_hgt(city_df, "/public/home/bedrock/work/geog/china_1_09km/geo_em.d01.nc")

    # 读取城市mask
    dcode = xr.open_dataset("/archive/share/projs/imes/default/static/mask.nc")["DCODE"].values

    # 人口数据
    pop = xr.open_dataset("/archive/share/Data/base/wgts/pop/eiss_popc_total.nc")["cnts"].values

    # 土地利用
    landuse_codes = {
        10: 'treecover', 20: 'shrubland', 30: 'grassland', 40: 'cropland',
        50: 'builtup', 60: 'bares', 70: 'snow', 80: 'water',
        90: 'wetland', 95: 'mangroves', 100: 'mossandlichen'
    }
    landuse_data = read_landuse_data(landuse_codes, "/archive/share/Data/base/wgts/lnd")

    # 计算结果
    result = []
    city_codes = np.unique(dcode)
    city_codes = city_codes[city_codes > 0]

    for city_code in city_codes:
        mask = (dcode == city_code)
        if not np.any(mask):
            continue

        city_info = city_df[city_df["code"] == city_code]
        if city_info.empty:
            continue

        city_name = city_info["cityName"].values[0]
        lon, lat, hgt = city_info[["lon", "lat", "hgt"]].values[0]
        population = np.nansum(pop[mask])

        landuse_result = {
            name: np.nanmean(landuse_data[name][mask]) / 10000.
            for name in landuse_codes.values()
        }

        row = {
            "code": int(city_code),
            "cityName": city_name,
            "lon": round(lon, 2),
            "lat": round(lat, 2),
            "hgt": round(hgt, 2),
            "area": round(area_dict.get(city_name, -999), 2),
            "population": int(round(population))
        }
        row.update({k: round(v, 5) for k, v in landuse_result.items()})
        result.append(row)

    # 保存结果
    result_df = pd.DataFrame(result)

    # 控制浮点数格式，避免科学计数法，全部保留小数点
    float_cols = ['lon', 'lat', 'hgt', 'area'] + list(landuse_codes.values())
    result_df[float_cols] = result_df[float_cols].astype(float)

    result_df.to_csv(
        "city_all.csv",
        index=False,
        encoding='utf-8',
        float_format='%.5f'  # 全部保留5位小数，不会科学计数法
    )
    print("CSV 已保存：city.csv")

if __name__ == "__main__":
    main()

