import sys
import pickle
from collections import deque
from datetime import datetime, timedelta

import torch
import numpy as np
import pandas as pd

from xformer import CFG
from xformer import XFormer, normalize, denormalize
from xformer import get_geo_infos, haversine_distance_matrix, get_x_and_y, geo_ngb_city_idx_by_batch

from preprocessing import read_all_data, LEAD_HOUR, site_fearture_file

def read_tmp_data(fileName, begTime, lead_hour, min_T, num_city):
    '''读取 历史观测数据 + 预报气象数据'''
    C_out = 6
    endTime = datetime(2024, 12, 31, 23)
    beg_idx = -(int((endTime - begTime).total_seconds() / 3600) + 1)
    end_idx = beg_idx + lead_hour
    beg_idx = (beg_idx - min_T) * num_city
    end_idx = end_idx * num_city
    data = pickle.load(open(fileName, 'rb'))[beg_idx:end_idx]
    data = data.to_numpy().reshape(-1, num_city, data.shape[1])

    data[..., :C_out] = np.log(data[..., :C_out] + 1e-4) # 对数化

    data[min_T:, :, :C_out] = -999.  # 未来污染物设为未知
    return data, C_out

def merge_data(data, static_data, geo_infos, k_neighbors=4):
    '''拼接静态数据与邻接信息'''
    # 计算风速和风向
    U10, V10 = data[..., 10], data[..., 11]
    wind_spd = np.sqrt(U10**2 + V10**2)
    wind_dir = (270 - np.degrees(np.arctan2(V10, U10))) % 360
    data[..., 10], data[..., 11] = wind_spd, wind_dir

    wind_spd = torch.tensor(wind_spd, dtype=torch.float32)
    wind_dir = torch.tensor(wind_dir, dtype=torch.float32)

    ngb_idx, ngb_wgt, ngb_arc, ngb_dst = geo_ngb_city_idx_by_batch(geo_infos, wind_spd, wind_dir, k_neighbors=k_neighbors, radius=500)

    data = torch.cat([torch.tensor(data, dtype=torch.float32), static_data.unsqueeze(0).expand(data.shape[0], -1, -1)], dim=-1)

    return [data, ngb_idx, ngb_wgt, ngb_arc, ngb_dst], data.shape[-1]

def run_prediction(model, dataset, x_mean, x_std, lead_hour, C_out, min_T, max_T, mete_only=False, poll_hour=24):
    '''运行逐步预报过程'''
    dataset[0] = normalize(dataset[0], x_mean, x_std)
    dataset = [x.unsqueeze(0) for x in dataset]  # [1, P, S, C]
    data, *ngb_info = dataset

    y_preds = deque(maxlen=1)
    y_s = []

    y_min = (x_mean[..., :C_out] * 0.1).unsqueeze(2)  # [1, 1, 1, 6]

    model.eval()
    with torch.no_grad():
        for t_step in range(lead_hour):
            x_batch, _, beg, end = get_x_and_y(data, C_out, t_step, y_preds, min_T, max_T)
            spatial_info = tuple(ngb[:, beg:end] for ngb in ngb_info)
            if t_step < poll_hour:
                y = model(x_batch, spatial_info, t_step=t_step, mete_only=mete_only)
                # y = model_mete(x_batch, spatial_info, t_step=t_step, mete_only=mete_only) # 用纯气象模型
            else:
                #y = model(x_batch, spatial_info, t_step=t_step, mete_only=True)
                y = model(x_batch, spatial_info, t_step=t_step, mete_only=mete_only)
            y_preds.append(y.detach())
            y_denorm = denormalize(y, x_mean[..., :C_out], x_std[..., :C_out])
            # y_denorm = torch.where(y_denorm < 0, y_min, y_denorm)
            y_denorm = torch.exp(y_denorm)  # 反对数化
            y_s.append(y_denorm.squeeze(0).squeeze(0))
    return torch.stack(y_s, dim=0).numpy()  # [lead_hour, S, C_out]

def write_output(y_s, begTime, site_file, city_file, out_file):
    '''输出站点级预报结果'''
    site_info = pd.read_csv(site_file, delim_whitespace=True)
    city_info = pd.read_csv(city_file)
    city_name_to_index = {row['cityName']: idx for idx, row in city_info.iterrows()}

    with open(begTime.strftime(out_file), 'w', encoding='utf-8') as f_out:
        for t in range(y_s.shape[0]):
            forecast_time = begTime + timedelta(hours=t)
            for _, site in site_info.iterrows():
                city_name = site['city']
                if city_name not in city_name_to_index:
                    pred_str = ', '.join("-999.00" for _ in range(y_s.shape[2]))
                else:
                    city_idx = city_name_to_index[city_name]
                    pred_str = ', '.join(f"{v:6.2f}" for v in y_s[t, city_idx, :])
                line = f"{(begTime - timedelta(hours=4)).strftime('%Y%m%d%H')},{forecast_time.strftime('%Y%m%d%H')},{site['code']},{pred_str}\n"
                f_out.write(line)

def main():
    begTime = datetime.strptime(sys.argv[1], "%Y%m%d%H")
    device = 'cpu'
    lead_hour = LEAD_HOUR

    k_neighbors = CFG.k_neighbors
    min_T, max_T = CFG.min_T, CFG.max_T

    model_file = CFG.mete_model if CFG.mete_only else CFG.poll_model
    checkpoint = torch.load(model_file, map_location=device)

    city_coords, static_data = get_geo_infos("./dataset/city.csv")
    city_coords = torch.tensor(city_coords, dtype=torch.float32)
    static_data = torch.tensor(static_data, dtype=torch.float32)
    geo_infos = {'coords': city_coords, 'dist_matrix': haversine_distance_matrix(city_coords)}
    
    C_out = 6
    data = read_all_data(begTime.strftime(site_fearture_file), checkpoint['x_mean'].numpy())

    # data, C_out = read_tmp_data("./dataset/city_data.pkl", begTime, lead_hour, min_T, static_data.shape[0])
    dataset, C_in = merge_data(data, static_data, geo_infos, k_neighbors=k_neighbors)

    model = XFormer(C_out, C_in - C_out, C_out, **CFG.kargs).to(device)
    model.load_state_dict(checkpoint['model_state_dict'])

    # checkpoint_mete = torch.load(CFG.mete_model, map_location=device)
    # model_mete = XFormer(C_out, C_in - C_out, C_out, **CFG.kargs).to(device)
    # model_mete.load_state_dict(checkpoint_mete['model_state_dict'])

    kargs = dict(min_T=min_T, max_T=max_T, mete_only=CFG.mete_only, poll_hour=CFG.poll_hour)
    y_s = run_prediction(model, dataset, checkpoint['x_mean'], checkpoint['x_std'], lead_hour, C_out, **kargs)

    write_output(
        y_s, begTime,
        site_file="./dataset/forecast_site.txt",
        city_file="./dataset/city.csv",
        out_file="./out/station_forecast_normal_hourly_xformer_d03_%Y%m%d.txt"
    )

if __name__ == "__main__":
    main()
