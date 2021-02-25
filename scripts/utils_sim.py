import cv2
import numpy as np

print('test')
def generate_aruco_markers(num):
    print('generating ', str(num), ' aruco markers')
    # load the ArUCo dictionary
    arucoDict = cv2.aruco.Dictionary_get(cv2.aruco.DICT_4X4_50)
    marker_size = 100
    for i in range(num):
        marker = np.zeros((marker_size, marker_size, 1), dtype="uint8")
        cv2.aruco.drawMarker(arucoDict, i, marker_size, marker, 1)
        # write the generated ArUCo tag to disk and then display it to our
        # screen
        cv2.imwrite('yarp/modules/marker_'+str(i)+'.png', marker)

if __name__ == "__main__":
    generate_aruco_markers(9)