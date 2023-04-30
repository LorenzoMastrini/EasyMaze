import math
import cv2
import numpy as np



src = np.zeros((256, 256, 1), dtype = "uint8")

cam = cv2.VideoCapture(0) # open the camera

captured = False
while True:
    ret, frame = cam.read()
    if not ret:
        print("failed to grab frame")
        break
    cv2.imshow("test", frame)

    k = cv2.waitKey(1)
    if k % 256 == 27: # ESC pressed
        print("Escape hit, closing...")
        break
    elif k % 256 == 32: # SPACE pressed
        captured = True
        src = frame
        print("Image captured successfully! Press ESC to close the camera.")
cam.release()
cv2.destroyAllWindows()

# Loads manually an image (in the same directory of this file)
if (not captured):
    src = cv2.imread('prova2.jpg')

src = cv2.cvtColor(src, cv2.COLOR_BGR2GRAY)

dst = cv2.Canny(src, 80, 200, None, 3)

# Copy edges to the images that will display the results in BGR
cdstP = cv2.cvtColor(dst, cv2.COLOR_GRAY2BGR)

linesP = cv2.HoughLinesP(dst, 1, np.pi / 180, 100, None, 10, 100)

if linesP is not None:
    for i in range(0, len(linesP)):
        l = linesP[i][0]
        if (math.isclose(l[0], l[2], rel_tol=0.3) or math.isclose(l[1], l[3], rel_tol=0.3)): # doesn't draw diagonal lines
            cv2.line(cdstP, (l[0], l[1]), (l[2], l[3]), (0,0,255), 5, cv2.LINE_AA)


#cv2.imshow("Detected Lines (in red) - Probabilistic Line Transform", cdstP)

result = cdstP.copy()
image = cv2.cvtColor(cdstP, cv2.COLOR_BGR2HSV)
lower = np.array([0,0,200])
upper = np.array([0,0,255])
mask = cv2.inRange(image, lower, upper)
result = cv2.bitwise_and(result, result, mask=mask)
kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3,3))
result = cv2.dilate(result, kernel, iterations=5)
result = cv2.erode(result, kernel, iterations=4)

cv2.namedWindow("Resized_Window", cv2.WINDOW_NORMAL)
#cv2.resizeWindow("Resized_Window", 300, 700)


#cv2.imshow('mask', mask)
cv2.imshow("Resized_Window", result)
cv2.waitKey(0)