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


def main_plotter_matplotlib():
    # plt.box(False)
    compartments_size = 20
    dataHM = pd.DataFrame(np.zeros((int(1440/compartments_size), int(2560/compartments_size))))
    resize_compartment_size = 5
    res_dataHM = pd.DataFrame(np.zeros((int(400/resize_compartment_size), int(400/resize_compartment_size))))
    # numpHM = np.zeros(shape=(int(1440/compartments_size), int(2560/compartments_size)))
    coll = data_Fetcher()
    print(len(coll))
    print(dataHM.shape)
    background_img = cv2.imread('imtest.jpg')
    IMG_ACTUAL_SIZE = background_img.shape

    IMG_RESIZE = (400,400)
    resize_bg_img = cv2.resize(background_img,IMG_RESIZE)
    poly_cords = [[750, 940], [1850, 420], [1980, 780], [960, 1280]]

    resized_poly_cords = co_ords_resizer(poly_cords,IMG_RESIZE, IMG_ACTUAL_SIZE)

    # xmin, ymin, xmax, ymax = get_around_box_coord(poly_cords)
    # print(xmin, ymin, xmax, ymax)

    for i in range(len(poly_cords)):
        poly_cords_X, poly_cords_Y = poly_cords[i]
        poly_cords[i] = [int(poly_cords_X/compartments_size), int(poly_cords_Y/compartments_size)]

        res_poly_cords_X, res_poly_cords_Y = resized_poly_cords[i]
        resized_poly_cords[i] = [int(res_poly_cords_X / resize_compartment_size), int(res_poly_cords_Y / resize_compartment_size)]
        # poly_cords[i] = int(poly_cords[i]/compartments_size)
        # resized_poly_cords[i] = int(resized_poly_cords[i] / resize_compartment_size)
    # xmin = int(xmin / compartments_size)
    # ymin = int(ymin / compartments_size)
    # xmax = int(xmax / compartments_size)
    # ymax = int(ymax / compartments_size)
    print('polycords and resized poly cords',poly_cords, resized_poly_cords)

    # xnew_min, ynew_min, xnew_max, ynew_max = get_around_box_coord(resized_poly_cords)
    # print(xnew_min, ynew_min, xnew_max, ynew_max)
    # xnew_min = int(xnew_min/resize_compartment_size)
    # ynew_min = int(ynew_min/resize_compartment_size)
    # xnew_max = int(xnew_max/resize_compartment_size)
    # ynew_max = int(ynew_max/resize_compartment_size)
    for one_dict in coll:
      for centroid in one_dict['BBoxes']:
        x = int(centroid[0]/compartments_size)
        y = int(centroid[1]/compartments_size)
        centroid_resize = co_ords_resizer([centroid], IMG_RESIZE, IMG_ACTUAL_SIZE)

        x_res = int(centroid_resize[0][0] / resize_compartment_size)
        y_res = int(centroid_resize[0][1] / resize_compartment_size)
        if(is_inside_polygon(poly_cords, (x,y))):
            dataHM[x][y] += 1

        if(is_inside_polygon(resized_poly_cords, (x_res,y_res))):
            res_dataHM[x_res][y_res] += 1
        # else:
        #     continue
        # numpHM[y][x] += 1
        # else:
        #     continue
    warped = four_point_transform(dataHM, poly_cords)



    plot1 = plt.imshow(dataHM, cmap='viridis', interpolation='bilinear', vmin=0, vmax=100)
    # print(x)
    plot1.axes.set_axis_off()


    # x.axes.patch.set_visible(False)
    # new = x.axes.get_images()
    plt.savefig('one.jpg',bbox_inches='tight',pad_inches = 0)
    heatmp = cv2.imread('one.jpg')
    res_img = cv2.resize(background_img, [heatmp.shape[1],heatmp.shape[0]])
    result_overlay = cv2.addWeighted(res_img, 0.5, heatmp, 0.7, 0)
    res = cv2.resize(result_overlay,(heatmp.shape[1]*3,heatmp.shape[0]*3), interpolation=cv2.INTER_LINEAR)
    cv2.imwrite('HeatMap1.jpg', res)

    plot2 = plt.imshow(res_dataHM, cmap='viridis', interpolation='bilinear', vmin=0, vmax=100)
    # print(x)
    plot2.axes.set_axis_off()
    plt.savefig('two.jpg', bbox_inches='tight', pad_inches=0)
    heatmp2 = cv2.imread('two.jpg')
    res_img2 = cv2.resize(resize_bg_img, [heatmp2.shape[1], heatmp2.shape[0]])
    result_overlay = cv2.addWeighted(res_img2, 0.5, heatmp2, 0.7, 0)
    res2 = cv2.resize(result_overlay, (heatmp2.shape[1] * 3, heatmp2.shape[0] * 3), interpolation=cv2.INTER_LINEAR)
    cv2.imwrite('HeatMap2.jpg', res2)

    squircled_img = to_square(resize_bg_img)
    res_img3 = cv2.resize(squircled_img, [heatmp2.shape[1], heatmp2.shape[0]])

    result_overlay = cv2.addWeighted(res_img3, 0.5, heatmp2, 0.7, 0)
    res2 = cv2.resize(result_overlay, (heatmp2.shape[1] * 3, heatmp2.shape[0] * 3), interpolation=cv2.INTER_LINEAR)
    cv2.imwrite('HeatMap3.jpg', res2)


    cv2.imwrite('WARPED.jpg', warped)
    cv2.imshow('Hello', warped)

    cv2.waitKey(0)
    # cv2.waitKey(0)
    # print(new)
    # plt.show()
    # plt.imshow(numpHM, cmap='viridis', interpolation='bilinear', vmin=0, vmax=100)
    # plt.savefig('two.png')
    # plt.show()

    resized_poly_cords = co_ords_resizer(poly_cords,IMG_RESIZE, IMG_ACTUAL_SIZE)
start_time = time.time()
main_plotter_matplotlib()
print(time.time() - start_time)
