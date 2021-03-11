import cv2
import numpy as np

def load_and_save(dataset):
    image_raw = np.load(dataset)
    # dataset.images_raw = np.load(folder + 'dataset_images.npy')

    print('dataset size size', image_raw.shape)
    #cv2.imshow('image_raw', image_raw[0])
    #cv2.imwrite('image_raw.png', image_raw[0])

if __name__ == "__main__":
    load_and_save('datasets/icub_alone/dataset_images_grayscale.npy')