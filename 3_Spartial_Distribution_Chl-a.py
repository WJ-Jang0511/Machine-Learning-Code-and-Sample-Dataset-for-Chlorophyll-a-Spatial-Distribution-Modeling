# -*- coding: utf-8 -*-
import os
import glob
import math
import pickle
import numpy as np
import pandas as pd
from osgeo import gdal, ogr
from scipy.ndimage import uniform_filter
from sklearn.preprocessing import MinMaxScaler

gdal.UseExceptions()

def GetValue(array, gt, Shp):
    ds = ogr.Open(Shp)
    lyr = ds.GetLayer()
    values = []

    for feat in lyr:
        geom = feat.GetGeometryRef()
        if geom.ExportToWkt()[:10] == "MULTIPOINT":
            Totxt = geom.ExportToWkt()[12:-1]
        else:
            Totxt = geom.ExportToWkt()[7:-1]

        Point = Totxt.split(",")[0]
        mx, my = float(Point.split()[0]), float(Point.split()[1])

        px = int((mx - gt[0]) / 2)
        py = int((my - gt[1]) / -2)

        values.append(float(array[py, px]))
    return values


def raster2array(File):
    metadata = {}
    dataset = gdal.Open(File)
    metadata['array_rows'] = dataset.RasterYSize
    metadata['array_cols'] = dataset.RasterXSize
    metadata['geotransform'] = dataset.GetGeoTransform()
    metadata['projection'] = dataset.GetProjection()

    raster = dataset.GetRasterBand(1)
    metadata['noDataValue'] = raster.GetNoDataValue()
    metadata['scaleFactor'] = raster.GetScale()

    mapinfo = dataset.GetGeoTransform()
    metadata['pixelWidth'] = mapinfo[1]
    metadata['pixelHeight'] = mapinfo[5]

    metadata['ext_dict'] = {}
    metadata['ext_dict']['xMin'] = mapinfo[0]
    metadata['ext_dict']['xMax'] = mapinfo[0] + dataset.RasterXSize * mapinfo[1]
    metadata['ext_dict']['yMin'] = mapinfo[3] + dataset.RasterYSize * mapinfo[5]
    metadata['ext_dict']['yMax'] = mapinfo[3]

    array = dataset.GetRasterBand(1).ReadAsArray(0, 0, metadata['array_cols'], metadata['array_rows']).astype(
        np.float64)

    return array, metadata


def array2raster(newRasterfn, array, metadata):
    cols = array.shape[1]
    rows = array.shape[0]

    driver = gdal.GetDriverByName('GTiff')
    outRaster = driver.Create(newRasterfn, cols, rows, 1, gdal.GDT_Float64)
    outRaster.SetGeoTransform((metadata["geotransform"][0], metadata["pixelWidth"], 0, metadata["geotransform"][3], 0,
                               metadata["pixelHeight"]))
    outRaster.SetProjection(metadata['projection'])
    outband = outRaster.GetRasterBand(1)

    if metadata['noDataValue'] is not None:
        outband.SetNoDataValue(np.nan)
    outband.SetNoDataValue(np.nan)
    outband.WriteArray(array)
    outband.FlushCache()


def listdivide(ar1, ar2):
    return [i / j for i, j in zip(ar1, ar2)]


def listmultiply(ar1, ar2):
    return [i * j for i, j in zip(ar1, ar2)]


def listinvert(ar1):
    return np.divide(np.ones((ar1.shape[0], ar1.shape[1])), ar1)


def listsum(ar1, ar2):
    return [i + j for i, j in zip(ar1, ar2)]


def listminus(ar1, ar2):
    return [i - j for i, j in zip(ar1, ar2)]


if __name__ == "__main__":

    Class = ['CL3']

    pkl = r"3CL_SMOTEENN_GB.pkl"

    Fname = os.path.basename(pkl).split(".")[0]

    Refdata = pd.read_csv(r"Sentinel_Sample\Ref.csv")
    X = Refdata[['B4/B6', 'B4/B8', 'B3/B6', 'B3/B7', 'B3/B8']].values

    Scaler = MinMaxScaler(feature_range=(0, 1))
    Scaler.fit(X)
    X_n = Scaler.transform(X)

    dates = ["20230616", "20230815"]

    for date in dates:
        cutlayer = r"Namyang.shp"
        Bandlist = ["B01", "B02", "B03", "B04", "B05", "B06", "B07", "B08", "B8A"]
        flist = []

        for Band in Bandlist:
            flist.append(glob.glob(r"Sentinel_Sample\*" + date + "_*" + Band + "*.tif")[0])

        resolutionneedtochange = ['B01', 'B05', 'B06', 'B07', 'B8A', 'B09', 'B11', 'B12']
        CLDNfile = r'CLDtemp.tif'

        gdal.Warp(CLDNfile, glob.glob(r"Sentinel_Sample\*" + date + "_CLDPRB.tif")[0],  cutlineDSName=abs_cutlayer, xRes=10, yRes=10, cropToCutline=True)

        CLDar, CLDmt = raster2array(CLDNfile)

        print(date)
        fdatas = []
        print("Input Generation")

        for file in flist:
            if file.split("\\")[-1].split("_")[-1].split(".")[0] in resolutionneedtochange:
                Nfile = '.\\temp.tif'
                gdal.Warp(Nfile, file, cutlineDSName=cutlayer, xRes=10, yRes=10, cropToCutline=True)
                ar, mt = raster2array(Nfile)
                fdatas.append([ar, mt])
            else:
                Nfile = '.\\temp.tif'
                gdal.Warp(Nfile, file, cutlineDSName=cutlayer, xRes=10, yRes=10, cropToCutline=True)
                ar, mt = raster2array(Nfile)
                fdatas.append([ar, mt])

        length = fdatas[3][0].shape[0] * fdatas[3][0].shape[1]
        size = [fdatas[3][0].shape[0], fdatas[3][0].shape[1]]
        meta = fdatas[3][1]

        if int(date[0:4]) >= 2022:
            B1 = compute_local_mean(fdatas[0][0]).reshape(length, 1) - 1000
            B2 = compute_local_mean(fdatas[1][0]).reshape(length, 1) - 1000
            B3 = compute_local_mean(fdatas[2][0]).reshape(length, 1) - 1000
            B4 = compute_local_mean(fdatas[3][0]).reshape(length, 1) - 1000
            B5 = compute_local_mean(fdatas[4][0]).reshape(length, 1) - 1000
            B6 = compute_local_mean(fdatas[5][0]).reshape(length, 1) - 1000
            B7 = compute_local_mean(fdatas[6][0]).reshape(length, 1) - 1000
            B8 = compute_local_mean(fdatas[7][0]).reshape(length, 1) - 1000
            B8A = compute_local_mean(fdatas[8][0]).reshape(length, 1) - 1000
        else:
            B1 = compute_local_mean(fdatas[0][0]).reshape(length, 1)
            B2 = compute_local_mean(fdatas[1][0]).reshape(length, 1)
            B3 = compute_local_mean(fdatas[2][0]).reshape(length, 1)
            B4 = compute_local_mean(fdatas[3][0]).reshape(length, 1)
            B5 = compute_local_mean(fdatas[4][0]).reshape(length, 1)
            B6 = compute_local_mean(fdatas[5][0]).reshape(length, 1)
            B7 = compute_local_mean(fdatas[6][0]).reshape(length, 1)
            B8 = compute_local_mean(fdatas[7][0]).reshape(length, 1)
            B8A = compute_local_mean(fdatas[8][0]).reshape(length, 1)

        NDWI = np.divide(np.subtract(B3, B8), B3 + B8)

        B1 = np.where(NDWI >= 0, B1, 0)
        B2 = np.where(NDWI >= 0, B2, 0)
        B3 = np.where(NDWI >= 0, B3, 0)
        B4 = np.where(NDWI >= 0, B4, 0)
        B5 = np.where(NDWI >= 0, B5, 0)
        B6 = np.where(NDWI >= 0, B6, 0)
        B7 = np.where(NDWI >= 0, B7, 0)
        B8 = np.where(NDWI >= 0, B8, 0)
        B8A = np.where(NDWI >= 0, B8A, 0)

        fdatas = []
        Input = np.hstack((np.divide(B4, B6),
                           np.divide(B4, B8),
                           np.divide(B3, B6),
                           np.divide(B3, B7),
                           np.divide(B3, B8)))

        Input = np.where(np.isinf(Input), 0, Input)
        Input = np.where(Input == -np.inf, 0, Input)
        Input = np.where(np.isnan(Input), 0, Input)

        Input = Scaler.transform(Input)
        print("Input Done")

        with open(pkl, 'rb') as file:
            Load_model = pickle.load(file)

        print(pkl + "Load")
        results = Load_model.predict(Input)

        print("Chl-a array done")
        results = results.reshape(size[0], size[1])
        base_ar = NDWI.reshape(size[0], size[1])

        results = np.where(results > 0, results, np.nan)
        results = np.where(CLDar <= 1, results, np.nan)

        array2raster(Fname + "_" + date + "_Chla.tif", results * 500, meta)