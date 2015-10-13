# File: Gesture_Detection.py
# Authors: Alex Casella, Raymond Chavez, Carlos Lovera
# Date: 2/13/14


import cv2
import numpy as np

# HSV values derived using the calibration trackbars below.
minH = 63
maxH = 255
minS = 136
maxS = 255
minV = 38
maxV = 157
trackbarWindowName = "Trackbars"
calibration = False
"""
H_MIN 63
H_MAX 255
S_MIN 136
S_MAX 255
V_MIN 38
V_MAX 157
"""

# HSV calibration trackbars.
def onSlide():
    setTrackbarVals()


def createTrackbars():
    cv2.namedWindow(trackbarWindowName)
    cv2.createTrackbar("H_MIN", trackbarWindowName, minH, 255, onSlide)
    cv2.createTrackbar("H_MAX", trackbarWindowName, maxH, 255, onSlide)
    cv2.createTrackbar("S_MIN", trackbarWindowName, minS, 255, onSlide)
    cv2.createTrackbar("S_MAX", trackbarWindowName, maxS, 255, onSlide)
    cv2.createTrackbar("V_MIN", trackbarWindowName, minV, 255, onSlide)
    cv2.createTrackbar("V_MAX", trackbarWindowName, maxV, 255, onSlide)


def setTrackbarVals():
    global minH, maxH, minS, maxS, minV, maxV
    minH = cv2.getTrackbarPos("H_MIN", trackbarWindowName)
    maxH = cv2.getTrackbarPos("H_MAX", trackbarWindowName)
    minS = cv2.getTrackbarPos("S_MIN", trackbarWindowName)
    maxS = cv2.getTrackbarPos("S_MAX", trackbarWindowName)
    minV = cv2.getTrackbarPos("V_MIN", trackbarWindowName)
    maxV = cv2.getTrackbarPos("V_MAX", trackbarWindowName)
    print("H_MIN " + str(minH))
    print("H_MAX " + str(maxH))
    print("S_MIN " + str(minS))
    print("S_MAX " + str(maxS))
    print("V_MIN " + str(minV))
    print("V_MAX " + str(maxV))


def main():
    cap = cv2.VideoCapture(0)
    # HSV calibration mode.
    if calibration:
        createTrackbars()

    # Main processing loop.
    while ( cap.isOpened() ):
        cv2.namedWindow("Gesture Detection")
        if calibration:
            setTrackbarVals()
        ( success, frame ) = cap.read()
        k = cv2.waitKey(10)
        if k == 27 or not success:
            print("Exiting.")
            break

        # Values for finding range of glove color in HSV.
        frameHSV = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        minHSV = np.array([minH, minS, minV], np.uint8)
        maxHSV = np.array([maxH, maxS, maxV], np.uint8)

        # Performing glove detection.
        gloveRegion = cv2.inRange(frameHSV, minHSV, maxHSV)
        blur = cv2.GaussianBlur(gloveRegion, (5, 5), 0)
        contours, hierarchy = cv2.findContours(blur, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # Finding largest contour.
        max_area = 0
        condrawing = np.zeros(frame.shape, np.uint8)
        for i in range(len(contours)):
            cnt = contours[i]
            area = cv2.contourArea(cnt)
            if (area > max_area):
                max_area = area
                ci = i

        # Storing largest contour, convex hull, and area of the hull.
        cnt = contours[ci]
        hull = cv2.convexHull(cnt)
        area = cv2.contourArea(hull)

        # If area is greater than 100,000, then detect five fingers.
        if (area > 100000):
            cv2.putText(condrawing, "Five fingers.", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 255), 2,
                        lineType=8)

        # If area is less than 100,000, but greater than 50,000, then detect two fingers.
        if (area < 100000 and area > 50000):
            cv2.putText(condrawing, "Two fingers.", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 255), 2,
                        lineType=8)

        # If area is less than 50,000, but greater than 10,000, then a closed fist.
        if (area < 50000 and area > 10000):
            cv2.putText(condrawing, "Closed fist.", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 255), 2,
                        lineType=8)

        # Draw the largest contour and the convex hull.
        cv2.drawContours(condrawing, [cnt], 0, (0, 140, 0), 2)
        cv2.drawContours(condrawing, [hull], 0, (0, 0, 255), 2)

        cv2.imshow("Gesture Detection", condrawing)
        cv2.imshow("Input", frame)


main()
