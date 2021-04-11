import cv2
import os
import numpy as np
import parameters
from sklearn.model_selection import train_test_split
from sklearn import preprocessing

# concatene the two datasets
def combine_datasets(DSa_folder, DSb_folder, combined_folder):
    if not os.path.exists(combined_folder):
        os.makedirs(combined_folder)

    a_timestamps = np.load(DSa_folder + 'dataset_timestamps.npy')
    b_timestamps = np.load(DSb_folder + 'dataset_timestamps.npy')
    combined_timestamps = np.hstack((a_timestamps, b_timestamps))
    print('combined_timestamps shape ', combined_timestamps.shape)
    np.save(combined_folder + 'dataset_timestamps.npy', combined_timestamps)

    a_joint_encoders = np.load(DSa_folder + 'dataset_joint_encoders.npy')
    b_joint_encoders = np.load(DSb_folder + 'dataset_joint_encoders.npy')
    combined_joint_encoders = np.vstack((a_joint_encoders, b_joint_encoders))
    print('combined_joint_encoders shape ', combined_joint_encoders.shape)
    np.save(combined_folder + 'dataset_joint_encoders.npy', combined_joint_encoders)

    a_motor_commands = np.load(DSa_folder + 'dataset_motor_commands.npy')
    b_motor_commands = np.load(DSb_folder + 'dataset_motor_commands.npy')
    combined_motor_commands = np.vstack((a_motor_commands, b_motor_commands))
    print('combined_motor_commands shape ', combined_motor_commands.shape)
    np.save(combined_folder + 'dataset_motor_commands.npy', combined_motor_commands)

    a_skin_values = np.load(DSa_folder + 'dataset_skin_values.npy')
    b_skin_values = np.load(DSb_folder + 'dataset_skin_values.npy')
    combined_skin_values = np.vstack((a_skin_values, b_skin_values))
    print('combined_skin_values shape ', combined_skin_values.shape)
    np.save(combined_folder + 'dataset_skin_values.npy', combined_skin_values)

    a_img = np.load(DSa_folder + 'dataset_images.npy')
    b_img = np.load(DSb_folder + 'dataset_images.npy')
    combined_img = np.vstack((a_img, b_img))
    print('combined_img shape ', combined_img.shape)
    np.save(combined_folder + 'dataset_images.npy', combined_img)

    a_img_grayscale = np.load(DSa_folder + 'dataset_images_grayscale.npy')
    b_img_grayscale = np.load(DSb_folder + 'dataset_images_grayscale.npy')
    combined_img_grayscale = np.vstack((a_img_grayscale, b_img_grayscale))
    print('combined_img_grayscale shape ', combined_img_grayscale.shape)
    np.save(combined_folder + 'dataset_images_grayscale.npy', combined_img_grayscale)



if __name__ == "__main__":
    combine_datasets('datasets/icub_alone/', 'datasets/icub_and_many_balls/', 'datasets/combined_alone_and_many_balls/')


#    param = parameters.Parameters()
#    img = np.load('datasets/icub_alone/dataset_images_grayscale.npy')
#    joints = np.load('datasets/icub_alone/dataset_joint_encoders.npy')
#    cmd = np.load('datasets/icub_alone/dataset_motor_commands.npy')