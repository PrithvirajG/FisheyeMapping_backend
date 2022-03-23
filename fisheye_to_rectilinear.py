import matplotlib.path as mplPath
import numpy as np
import cv2
import pandas as pd
import matplotlib.pyplot as plt

def merge_mapper(x, y):
    k1 = 2*x + 22 - y

    k2 = 2*x - y
    print(x,y)
    if(k1*k2 < 0):
        return 1
    else:
        return 0

def get_around_box_coord(polycord):
    xmax = 0
    ymax = 0
    xmin = 9000
    ymin = 9000
    for coord in polycord:
        x,y = coord
        if x>xmax:
            xmax = x
        if x<xmin:
            xmin = x
        if y>ymax:
            ymax = y
        if y<ymin:
            ymin = y
    return xmin, ymin, xmax, ymax
        # listx.append(x)
        # listy.append(y)

def is_inside_polygon(poly_cords, point):
    poly_path = mplPath.Path(np.array(poly_cords))
    if(poly_path.contains_point(point)):
        return 1
    else:
        return 0

def box_around_poly(xcurr, ycurr, xmax, ymax, xmin,  ymin):
    if((xcurr > xmax) or (xcurr < xmin) or (ycurr>ymax) or (ycurr<ymin)):
        return 0
    else:
        return 1

def co_ords_resizer(polycords, RESIZEIMG, ACTUALIMGSIZE):
    new_polycords = []
    for cord in polycords:
        x, y = cord
        x_new = (x/ACTUALIMGSIZE[1])*RESIZEIMG[1]
        y_new = (y / ACTUALIMGSIZE[0]) * RESIZEIMG[0]
        new_polycords.append([x_new,y_new])
        # print(new_polycords)
    return new_polycords

def four_point_transform(image, pts, zoneid):
    # obtain a consistent order of the points and unpack them
    # individually
    # normalized_df = (image - image.min()) / (image.max() - image.min())
    # print(normalized_df)
    # print(normalized_df.max(), normalized_df.min())


    pts_np = np.array(pts)
    rect = order_points(pts_np)
    (tl, tr, br, bl) = rect

    # compute the width of the new image, which will be the
    # maximum distance between bottom-right and bottom-left
    # x-coordiates or the top-right and top-left x-coordinates
    widthA = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
    widthB = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
    maxWidth = max(int(widthA), int(widthB))

    # compute the height of the new image, which will be the
    # maximum distance between the top-right and bottom-right
    # y-coordinates or the top-left and bottom-left y-coordinates
    heightA = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
    heightB = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
    maxHeight = max(int(heightA), int(heightB))

    # now that we have the dimensions of the new image, construct
    # the set of destination points to obtain a "birds eye view",
    # (i.e. top-down view) of the image, again specifying points
    # in the top-left, top-right, bottom-right, and bottom-left
    # order
    dst = np.array([
        [0, 0],
        [maxWidth - 1, 0],
        [maxWidth - 1, maxHeight - 1],
        [0, maxHeight - 1]], dtype = "float32")

    # compute the perspective transform matrix and then apply it
    M = cv2.getPerspectiveTransform(rect, dst)
    img_np = image.to_numpy()
    warped = cv2.warpPerspective(img_np, M, (maxWidth, maxHeight))

    # warped = cv2.resize(warped,(500,500))
    plotw = plt.imshow(warped, cmap='viridis', interpolation='bilinear', vmin=0, vmax=100)
    plotw.axes.set_axis_off()

    # x.axes.patch.set_visible(False)
    # new = x.axes.get_images()
    plt.savefig('Zones/zone'+ str(zoneid) +'.jpg', bbox_inches='tight', pad_inches=0)
    zone_heatmap =  cv2.imread('Zones/zone'+ str(zoneid) +'.jpg')
    # cv2.imshow('Yohhooho', plotw.to_numpy())

    # return the warped image
    return zone_heatmap

def order_points(pts):

	# initialzie a list of coordinates that will be ordered
	# such that the first entry in the list is the top-left,
	# the second entry is the top-right, the third is the
	# bottom-right, and the fourth is the bottom-left
	rect = np.zeros((4, 2), dtype = "float32")
	# the top-left point will have the smallest sum, whereas
	# the bottom-right point will have the largest sum
	s = pts.sum(axis = 1)
	rect[0] = pts[np.argmin(s)]
	rect[2] = pts[np.argmax(s)]
	# now, compute the difference between the points, the
	# top-right point will have the smallest difference,
	# whereas the bottom-left will have the largest difference
	diff = np.diff(pts, axis = 1)
	rect[1] = pts[np.argmin(diff)]
	rect[3] = pts[np.argmax(diff)]
	# return the ordered coordinates
	return rect