import pandas as pd

df_city = pd.read_csv('city.csv')
df_site = pd.read_csv('site.txt', sep='\s+', encoding='utf-8')
#df_site = pd.read_csv('forecast_site.txt', sep='\s+', encoding='utf-8')

site_cities = set(df_site['city'].unique())
city_cities = set(df_city['cityName'].unique())

print("❗以下城市名在 site.txt 中存在，但 city.csv 中没有：")
for city in sorted(site_cities - city_cities):
    print(city)


print("❗以下城市名在 city.txt 中存在，但 site.csv 中没有：")
for city in sorted(city_cities - site_cities):
    print(city)
