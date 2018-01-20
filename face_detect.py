from imutils import face_utils
import numpy as np
import argparse
import imutils
import dlib
import cv2
from matplotlib.path import Path

# code from https://www.pyimagesearch.com/2017/04/03/facial-landmarks-dlib-opencv-python/

filename = 'cameron.mp4'


cap = cv2.VideoCapture(filename)


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

def get_frame_at(seconds, cap, fps):
    frame_num = int(seconds * fps)
    cap.set(2, frame_num)
    ret, image = cap.read()
    return (ret, image)

def face_detect(image, detector, predictor):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # detect faces in the grayscale image
    rects = detector(gray, 1)
    for (i, rect) in enumerate(rects):
        shape = predictor(gray, rect)
    return rects, shape

def get_left_cheek_points(shape):
    polygon = [shape[2], shape[41], shape[48]]
    return get_roi(polygon)

def get_right_cheek_points(shape):
    polygon = [shape[46], shape[14], shape[54]]
    return get_roi(polygon)

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
    values = image[masked_points[:,0], masked_points[:,1], :]
    means = np.mean(values, 0)
    return means



# initialize dlib's face detector (HOG-based) and then create
# the facial landmark predictor
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

# load the input image, resize it, and convert it to grayscale
ret = True
while(ret == True):
    ret, image = cap.read()
    print(type(image))
    #image = cv2.imread(frame)
    # image = imutils.resize(image, width=500)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # detect faces in the grayscale image
    rects = detector(gray, 1)

    for (i, rect) in enumerate(rects):
        # determine the facial landmarks for the face region, then
        # convert the facial landmark (x, y)-coordinates to a NumPy
        # array
        shape = predictor(gray, rect)
        shape = face_utils.shape_to_np(shape)

        # convert dlib's rectangle to a OpenCV-style bounding box
        # [i.e., (x, y, w, h)], then draw the face bounding box
        (x, y, w, h) = face_utils.rect_to_bb(rect)
#        print(x, y, w, h)
        cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)

        # loop over the (x, y)-coordinates for the facial landmarks
        # and draw them on the image
        masked = get_right_cheek_points(shape)
        foo = get_average_in_roi(masked, image)
        for index, (x, y) in enumerate(masked):
            if x != None and y != None:
                cv2.circle(image, (x, y), 1, (255, 255, 255), -1)

    # show the output image with the face detections + facial landmarks
    cv2.imshow("Output", image)
    cv2.waitKey(0)


