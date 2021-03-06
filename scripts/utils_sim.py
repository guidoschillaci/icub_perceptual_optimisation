import cv2
import numpy as np
from copy import deepcopy

class arucoDetector():
    def __init__(self):
        self.arucoParams = cv2.aruco.DetectorParameters_create()
        self.arucoDict = cv2.aruco.Dictionary_get(cv2.aruco.DICT_4X4_50)
        self.marker_size = 100
        self.border_size = 10

    def generate_aruco_markers(self, num):
        print('generating ', str(num), ' aruco markers')
        # load the ArUCo dictionary

        for i in range(num):
            marker = np.zeros((self.marker_size, self.marker_size, 1), dtype="uint8")
            cv2.aruco.drawMarker(self.arucoDict, i, self.marker_size, marker, 1)

            white = [255,255,255]
            constant= cv2.copyMakeBorder(marker,self.border_size,self.border_size,self.border_size,self.border_size,cv2.BORDER_CONSTANT,value=white)

            # write the generated ArUCo tag to disk and then display it to our
            # screen
            cv2.imwrite('yarp/data/markers/marker_'+str(i)+'.png', constant)
            #cv2.imwrite('yarp/data/markers/marker_'+str(i)+'.bmp', constant)

    # count how many aruco markers are in the given image
    def count_aruco_markers_from_single_image(self, img):
        height, width = img.shape
        if height != 320 or width != 240:
            cv2_img = cv2.resize(img, (320,240))
        else:
            cv2_img = deepcopy(img)
        #print("detecting aruco markers")
        #cv2.imwrite('marker_img.png',cv2_img)

        (corners, ids, rejected) = cv2.aruco.detectMarkers(cv2_img, self.arucoDict, parameters=self.arucoParams)
        ids_sorted =np.sort(ids)
        #print('detected ids ', str(np.asarray(ids_sorted)))
        return len(ids_sorted), ids_sorted

    # returns the average number of markers detected in a given list of images
    def avg_mrk_in_list_of_img(self, list_img):
        num_markers_in_img = 0
        for i in range(len(list_img)):
            num_markers, id_markers = self.count_aruco_markers_from_single_image(list_img[i])
            num_markers_in_img += num_markers
        return num_markers_in_img / len(list_img)


if __name__ == "__main__":
    _arucoDetector = arucoDetector()
    img = cv2.imread("datasets/background_image.png")
    # generate_aruco_markers(9,arucoDict)
    _arucoDetector.count_aruco_markers_from_single_image(img)

