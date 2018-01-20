# TODO : work for multiple faces.

from imutils import face_utils
import numpy as np
import argparse
import imutils
import dlib
import cv2
from matplotlib.path import Path

# code from https://www.pyimagesearch.com/2017/04/03/facial-landmarks-dlib-opencv-python/



def rect_to_bb(rect):
    # take a bounding predicted by dlib and convert it
    # to the format (x, y, w, h) as we would normally do
    # with OpenCV
    x = rect.left()
    y = rect.top()
    w = rect.right() - x
    h = rect.bottom() - y

    # return a tuple of (x, y, w, h)
    return (x, y, w, h)

def shape_to_np(shape, dtype="int"):
    # initialize the list of (x, y)-coordinates
    coords = np.zeros((68, 2), dtype=dtype)

    # loop over the 68 facial landmarks and convert them
    # to a 2-tuple of (x, y)-coordinates
    for i in range(0, 68):
        coords[i] = (shape.part(i).x, shape.part(i).y)

    # return the list of (x, y)-coordinates
    return coords

def get_frame_at(frame_num, cap):
    frame_num = int(frame_num)
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_num)
    ret, image = cap.read()
    return (ret, image)

def face_detect(image, detector, predictor):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # detect faces in the grayscale image
    rects = detector(gray, 1)
    shapes = []
    for (i, rect) in enumerate(rects):
        shape = predictor(gray, rect)
        shape = shape_to_np(shape)
        shapes.append(shape)
    return rects, shapes

def get_left_cheek_points(shape):
    polygon = [shape[2], shape[41], shape[48]]
    return get_roi(polygon), polygon

def get_right_cheek_points(shape):
    polygon = [shape[46], shape[14], shape[54]]
    return get_roi(polygon), polygon

def get_roi(polygon):
    justy = np.array(polygon)[:,1]
    justx = np.array(polygon)[:,0]
    miny = justy.min()
    minx = justx.min()
    maxy = justy.max()
    maxx = justx.max()
    poly = []
    for element in polygon:
        poly.append((element[0], element[1]))
    # shape[31] could also be included but it's the side of the nose which doesn't help much??
    x, y = np.meshgrid(np.arange(minx, maxx), np.arange(miny, maxy)) # make a canvas with coordinates
    x, y = x.flatten(), y.flatten()
    points = np.vstack((x,y)).T
    p = Path(poly) # make a polygon
    grid = p.contains_points(points)
    grid = np.vstack((grid, grid)).T
    grid = np.invert(grid)
    masked_points = np.ma.masked_array(points, grid)
    return masked_points

def get_average_in_roi(masked_points, image):
    # THE IMAGE IS IN BGR NOT RGB!
    values = image[masked_points[:,1].compressed(), masked_points[:,0].compressed(), :]
    means = np.mean(values, 0)
    return means

def get_frame_average(frame,shape, points_func):
    average = None
    # CHANGE THIS TO SUPPORT MORE FACES LATER
    points, polygon = points_func(shape)
    if len(points) > 0:
        average = get_average_in_roi(points,frame)
    # for x, y in points:
    #         if x != None and y != None:
    #             cv2.circle(frame, (x, y), 1, (average[0], average[1], average[2]), -1)
    # if flag == True:

    #     # show the output image with the face detections + facial landmarks
    #     cv2.imshow("Output", frame)
    #     cv2.waitKey(0)

    # print(average)
    return average



def get_time_series(cap,start,frames,freq):
    fps = cap.get(cv2.CAP_PROP_FPS)
    end = start + frames * fps/freq
    frame_list = np.round(np.arange(start,end,fps/freq))
    print(frame_list)
    print(len(frame_list))
    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")
    left_cheeky = []
    right_cheeky = []
    for frame in frame_list:
        print(frame)
        ret, fr = get_frame_at(frame,cap)
        rects, shapes = face_detect(fr,detector,predictor)
        if shapes != []:
            for shape in shapes:
                left_cheeky.append(get_frame_average(fr,shape,get_left_cheek_points))
                right_cheeky.append(get_frame_average(fr,shape, get_right_cheek_points))
            if type(left_cheeky[-1]) == type(None):
                left_cheeky[-1] = left_cheeky[-2]
            if type(right_cheeky[-1]) == type(None):
                right_cheeky[-1] = right_cheeky[-2]
        else:
            left_cheeky.append(left_cheeky[-1])
            right_cheeky.append(right_cheeky[-1])
            print("Duplicated")
    return(np.array(left_cheeky), np.array(right_cheeky))




filename = 'sj.mp4'


cap = cv2.VideoCapture(filename)


l, r = get_time_series(cap,0,500,24)
# print(l)
np.save("left_sj.npy", l)
np.save("right_sj.npy", r)

# detector = dlib.get_frontal_face_detector()
# predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")
# # initialize dlib's face detector (HOG-based) and then create
# # the facial landmark predictor

# # load the input image, resize it, and convert it to grayscale
# ret = True
# while(ret == True):
#     ret, image = cap.read()
#     print(type(image))
#     #image = cv2.imread(frame)
#     # image = imutils.resize(image, width=500)
#     gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

#     # detect faces in the grayscale image
#     rects, shapes = face_detect(image, detector, predictor)

#     for (i, rect) in enumerate(rects):
#         # determine the facial landmarks for the face region, then
#         # convert the facial landmark (x, y)-coordinates to a NumPy
#         # array
#         shape = shapes[i]

#         # convert dlib's rectangle to a OpenCV-style bounding box
#         # [i.e., (x, y, w, h)], then draw the face bounding box
#         (x, y, w, h) = face_utils.rect_to_bb(rect)
# #        print(x, y, w, h)
#         cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)

#         # loop over the (x, y)-coordinates for the facial landmarks
#         # and draw them on the image
#         masked = get_right_cheek_points(shape)
#         foo = get_average_in_roi(masked, image)
#         for index, (x, y) in enumerate(masked):
#             if x != None and y != None:
#                 cv2.circle(image, (x, y), 1, (255, 255, 255), -1)

#     # show the output image with the face detections + facial landmarks
#     cv2.imshow("Output", image)
#     cv2.waitKey(0)


