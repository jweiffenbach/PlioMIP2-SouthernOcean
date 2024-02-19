import numpy as np
import xarray as xr


def lon180(ds):
    "Rewrite 0-360 longitude to -180-180"
    ds.coords['lon'] = (ds.coords['lon'] + 180) % 360 - 180
    ds = ds.sortby(ds.lon)
    return ds

def mmm(variable, minmod = 5):
    mmm = variable.where(variable>-1e10).where(variable<1e10).where(variable.count(dim='model')>=minmod).mean(dim='model')
    return mmm

def smm(variable, minmod = 3):
    smm = variable.where(variable>-1e10).where(variable<1e10).where(variable.count(dim='model')>=minmod).mean(dim='model')
    return smm

def stad(variable, minmod = 3):
    std = variable.where(variable>-1e10).where(variable<1e10).where(variable.count(dim='model')>=minmod).std(dim='model')
    return std

def siedge(si):
    zm = si.mean(dim='lon')
    edge = zm.where(zm.lat<0,drop=True).where(zm>=15, drop=True).lat.max(dim='lat')
    return (edge.values)


def mask(dsE280, dsEoi400, mmmEoi400, threshold=12):
    #Arrays that count how many models show the same + or - sign for Eoi400-E280
    count_pos = mmmEoi400.where((dsEoi400-dsE280)>0).count(dim='model')
    count_neg = mmmEoi400.where((dsEoi400-dsE280)<0).count(dim='model')

    #Construct 2D arrays of longitude and latitude to produce scatter plots for significance (>= 12 models agree on sign)
    lon = dsE280.lon
    lat = dsE280.lat

    coords = xr.Dataset(data_vars = dict(lat2D=(["lat", "lon"], np.repeat(lat.values[:, np.newaxis], len(lon), axis=1))), coords = dict(lat = lat.values, lon = lon.values))
    coords['lon2D'] = coords.lat2D.copy(deep=True)
    coords.lon2D[:] = np.transpose(np.repeat(lon.values[:, np.newaxis], len(lat), axis=1))

    mask_pos = count_pos.where(count_pos>=threshold).fillna(0)
    mask_neg = count_neg.where(count_neg>=threshold).fillna(0)
    
    mask = mask_pos+mask_neg
    
    return(coords, mask)


def maskzonmean(dsE280, dsEoi400, mmmEoi400, threshold=12):
    #Arrays that count how many models show the same + or - sign for Eoi400-E280
    count_pos = mmmEoi400.where((dsEoi400-dsE280)>0).count(dim='model')
    count_neg = mmmEoi400.where((dsEoi400-dsE280)<0).count(dim='model')

    #Construct 2D arrays of longitude and latitude to produce scatter plots for significance (>= 12 models agree on sign)
    lat=dsE280.lat
    z = dsE280.z

    coords = xr.Dataset(data_vars = dict(z2D=(["z", "lat"], np.repeat(z.values[:, np.newaxis]/1000, len(lat), axis=1))), coords = dict(z = z.values/1000, lat = lat.values))
    coords['lat2D'] = coords.z2D.copy(deep=True)
    coords.lat2D[:] = np.transpose(np.repeat(lat.values[:, np.newaxis], len(z), axis=1))

    mask_pos = count_pos.where(count_pos>=threshold).fillna(0)
    mask_neg = count_neg.where(count_neg>=threshold).fillna(0)
    
    mask = mask_pos+mask_neg
    mask['z']=mask.z/1000
    return(mask)