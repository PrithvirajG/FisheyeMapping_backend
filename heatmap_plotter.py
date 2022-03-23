import time

import cv2
from DB.mongo_connection import get_mongo_client
from loguru import logger
from fisheye_to_rectilinear import merge_mapper, box_around_poly,get_around_box_coord,co_ords_resizer, is_inside_polygon,four_point_transform
import numpy as np
from datetime import datetime
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from squircle import to_circle, to_square
# %matplotlib inline
def ConnectMongo():
    while True:
        mongo_coll_ret = get_mongo_client()
        # meta_data = get_meta_data()
        if mongo_coll_ret:
            logger.info("Connected to MongoDB successfully.")
            break
    # meta_data_dict_ret = meta_data.find()
    return mongo_coll_ret


def convert_Date_to_int(date):
    date_ret = date[0:10]
    date_YYYY = date[0:4]
    date_MM = date[5:7]
    date_DD = date[8:10]
    date_int = int(date_YYYY) * 10000 + int(date_MM) * 100 + int(date_DD)
    return date_int


def convert_Time_to_int(time):
    date_ret = time[0:10]
    time_HH = time[0:2]
    time_MM = time[3:5]
    time_SS = time[6:8]
    time_int = int(time_HH) * 10000 + int(time_MM) * 100 + int(time_SS)
    return time_int


def Filter_From_and_To_Datetime(read_date_time):
    start_date_time = datetime(2022, 3, 8,5, 30, 0)
    end_date_time = datetime(2022, 3, 8,12, 30, 0)
    returncode = 0
    if (read_date_time >= start_date_time and read_date_time <= end_date_time):
        returncode = 1
    else:
        returncode
    return returncode


def data_Fetcher():
    mongoColl = ConnectMongo()
    coll = mongoColl.find()
    #print("Row:",coll)
    time_filtered_list = []
    for one_dict in coll:
        #print(one_dict)
        #print(one_dict["First_Found_Read_Time"])
        x = Filter_From_and_To_Datetime(one_dict["First_Found_Read_Time"])
        if (x):
            #print("TRUE", one_dict["First_Found_Read_Time"])
            time_filtered_list.append(one_dict)
        else:
            continue
    return time_filtered_list


def Filter_ID_specific_points(time_filt_list):
    pass


def main_plotter():
    RGB_COLOR = (65, 0, 85)
    COMPARTMENT_SIZE = 20

    ZONES = [[605, 190, 705, 530],
             [545, 550, 840, 617],
             [730, 331, 883, 520],
             [730, 190, 883, 331]
             ]
    QUAD_CORDS = [[[750, 940], [1850, 420], [1980, 780], [960, 1280]],
                  [[1565, 0], [1995, 1], [2410, 1000], [2022, 901]],
                  [[1560, 0], [1660, 185], [1170, 420], [930, 0]],
                  [[925, 0], [1165, 420], [617, 684], [557, 277]]]

    dataHM = pd.DataFrame(np.zeros((int(1440 / COMPARTMENT_SIZE), int(2560 / COMPARTMENT_SIZE))))


    coll = data_Fetcher()
    print(len(coll))
    print(dataHM.shape)
    background_img = cv2.imread('imtest.jpg')
    IMG_ACTUAL_SIZE = background_img.shape

    floormap_image = cv2.imread('FloorMap.jpeg')
    floorMapHM = np.zeros((floormap_image.shape[0], floormap_image.shape[1], 3))

    color = tuple(reversed(RGB_COLOR))
    floorMapHM[:] = color

    for i in range(len(QUAD_CORDS)):
        for j in range(4):
            QUAD_CORDS_X, QUAD_CORDS_Y = QUAD_CORDS[i][j]
            QUAD_CORDS[i][j] = [int(QUAD_CORDS_X / COMPARTMENT_SIZE), int(QUAD_CORDS_Y / COMPARTMENT_SIZE)]
    print('polycords', QUAD_CORDS)

    for one_dict in coll:
        for centroid in one_dict['BBoxes']:
            x = int(centroid[0] / COMPARTMENT_SIZE)
            y = int(centroid[1] / COMPARTMENT_SIZE)

            for i in range(len(QUAD_CORDS)):
                if (is_inside_polygon(QUAD_CORDS[i], (x, y))):
                    dataHM[x][y] += 1
            # else:
            #     continue
            # numpHM[y][x] += 1
            # else:
            #     continue

    # for zone in ZONES:

    for i in range(len(QUAD_CORDS)):
        zoneid = i
        zone_heatmap = four_point_transform(dataHM, QUAD_CORDS[i], zoneid)
        zone_heatmap = np.rot90(zone_heatmap, 3)
        zone_heatmap_res = cv2.resize(zone_heatmap, (abs(ZONES[i][0] - ZONES[i][2]), abs(ZONES[i][1] - ZONES[i][3])))
        floorMapHM[ZONES[i][1]:ZONES[i][1] + zone_heatmap_res.shape[0],
        ZONES[i][0]:ZONES[i][0] + zone_heatmap_res.shape[1]] = zone_heatmap_res
        result_overlay = cv2.addWeighted(floormap_image, 0.5, floorMapHM, 0.7, 0, dtype=cv2.CV_64F)
        cv2.imwrite('HeatmapZone' + str(i) + '.jpg', zone_heatmap_res)
        cv2.imwrite('HeatmapOnFloormap' + str(i) + '.jpg', floorMapHM)
        cv2.imwrite('FinalPlotted' + str(i) + '.jpg', result_overlay)

        # cv2.imshow('Final', result_overlay)
        # cv2.imshow('asdsada', floormap_image)
        # cv2.imshow('sdssd', zone_heatmap_res)
        # cv2.waitKey(0)
    # cv2.rotate(zone_heatmap)
    # print(type(zone_heatmap))

    print(floorMapHM.shape , floormap_image.shape)


    plotw = plt.imshow(dataHM, cmap='viridis', interpolation='bilinear', vmin=0, vmax=100)
    plotw.axes.set_axis_off()
    plt.savefig('FISHEYE_PLOT.jpg', bbox_inches='tight', pad_inches=0)
    # cv2.imshow('FInalll',dataHM.to_numpy())
    # cv2.waitKey(0)
    # print(type(floormap_image))
    # result_overlay = cv2.addWeighted(floormap_image, 0.5, zone_heatmap, 0.7, 0)
    # plt.savefig('one.jpg', bbox_inches='tight', pad_inches=0)
    heatmp = cv2.imread('FISHEYE_PLOT.jpg')
    res_img = cv2.resize(background_img, [heatmp.shape[1], heatmp.shape[0]])
    result_overlay_fisheye = cv2.addWeighted(res_img, 0.5, heatmp, 0.7, 0)
    cv2.imwrite('FISHEYE_PLOT_OVERLAY.jpg', result_overlay_fisheye)


main_plotter()
