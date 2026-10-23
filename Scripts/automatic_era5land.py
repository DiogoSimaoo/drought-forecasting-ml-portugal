import cdsapi


start_year = 2022
end_year = 2022
client = cdsapi.Client()

for year in range(start_year, end_year + 1):
    for month in range(1, 13):   
        dataset = "reanalysis-era5-land"
        request = {
            "variable": ["2m_dewpoint_temperature"],
            "year": str(year),
            "month": f"{month:02d}",  
            "day": [f"{day:02d}" for day in range(1, 32)],  
            "time": [f"{hour:02d}:00" for hour in range(24)],  
            "data_format": "netcdf",
            "download_format": "unarchived",
            "area": [42.16, -9.56, 36.95, -6.19], 
        }

        
        filename = f"ERA5_{year}_{month:02d}_2mdt.nc"

        
        print(f"Baixando dados para {year}-{month:02d}...")
        client.retrieve(dataset, request).download(filename)

print("Download completo!")
