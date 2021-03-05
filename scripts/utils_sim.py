import cv2
import numpy as np

def generate_aruco_markers(num, arucoDict):
    print('generating ', str(num), ' aruco markers')
    # load the ArUCo dictionary
    
    marker_size = 100
    for i in range(num):
        marker = np.zeros((marker_size, marker_size, 1), dtype="uint8")
        cv2.aruco.drawMarker(arucoDict, i, marker_size, marker, 1)

        white = [255,255,255]
        border_size = 10
        constant= cv2.copyMakeBorder(marker,border_size,border_size,border_size,border_size,cv2.BORDER_CONSTANT,value=white)

        # write the generated ArUCo tag to disk and then display it to our
        # screen
        cv2.imwrite('yarp/data/markers/marker_'+str(i)+'.png', constant)
        #cv2.imwrite('yarp/data/markers/marker_'+str(i)+'.bmp', constant)


def read_aruco_markers(img, arucoDict = cv2.aruco.DICT_4X4_50):
    cv2_img = cv2.resize(img, (320,240))
    print("detecting aruco markers")
    cv2.imwrite('marker_img.png',cv2_img)
    arucoParams = cv2.aruco.DetectorParameters_create()
    (corners, ids, rejected) = cv2.aruco.detectMarkers(cv2_img, arucoDict, parameters=arucoParams)
    ids_sorted =np.sort(ids.flatten())
    print('detected ids ', str(np.asarray(ids_sorted))) 
    return ids_sorted, len(ids_sorted)        


if __name__ == "__main__":
    arucoDict = cv2.aruco.Dictionary_get(cv2.aruco.DICT_4X4_50)
    img = cv2.imread("datasets/background_image.png")
    # generate_aruco_markers(9,arucoDict)
    read_aruco_markers(img, arucoDict)

