import cv2
import numpy as np
import parameters
from sklearn.model_selection import train_test_split
from sklearn import preprocessing

def load_and_save(dataset):
    image_raw = np.load(dataset)
    # dataset.images_raw = np.load(folder + 'dataset_images.npy')

    print('dataset size size', image_raw.shape)
    #cv2.imshow('image_raw', image_raw[0])
    #cv2.imwrite('image_raw.png', image_raw[0])

def check_mean(dataset):
    meann = np.mean(dataset, axis=0)
    print('mean ',str(np.mean(meann)))
    pass

def check_std(dataset):
    stdd = np.std(dataset, axis=0)
    #print('stddev ',str(stdd))
    print('mean stddev ',str(np.mean(stdd)))


if __name__ == "__main__":
    param = parameters.Parameters()
    img = np.load('datasets/icub_alone/dataset_images_grayscale.npy')
    joints = np.load('datasets/icub_alone/dataset_joint_encoders.npy')
    cmd = np.load('datasets/icub_alone/dataset_motor_commands.npy')
    #load_and_save('datasets/icub_alone/dataset_images_grayscale.npy')
    scaler_dataset_joints = preprocessing.MinMaxScaler()
    joints = scaler_dataset_joints.fit_transform(joints)
    scaler_dataset_motor = preprocessing.MinMaxScaler()
    cmd = scaler_dataset_motor.fit_transform(cmd)

    index_from = len(cmd) - int(len(cmd) * param.get('test_dataset_factor')) - 1




    _, joint_shuffled = train_test_split(joints, \
                            test_size =param.get('test_dataset_factor'), \
                            random_state = param.get('dataset_split_seed'))
    _, cmd_shuffled = train_test_split(cmd, \
                            test_size =param.get('test_dataset_factor'), \
                            random_state = param.get('dataset_split_seed'))

    img1 = cv2.resize(img[(index_from+128)],
                    (param.get('image_size'), param.get('image_size')),
                                 interpolation=cv2.INTER_LINEAR)
    img2 = cv2.resize(img[(index_from+129)],
                    (param.get('image_size'), param.get('image_size')),
                                 interpolation=cv2.INTER_LINEAR)
    cv2.imwrite('image_raw.png', img1)
    cv2.imwrite('image_raw_2.png', img2)
    flow = cv2.calcOpticalFlowFarneback(img1, img2, None, 0.5, 3, 5, 3, 5, 0.9, 0)
    magnitude, ang = cv2.cartToPolar(flow[..., 0], flow[..., 1])
    thresh = np.array(np.where(magnitude > param.get('opt_flow_binary_threshold'), 1, 0), dtype='uint8')
    print('non zeros ', np.count_nonzero(thresh == 1))
    print('list ', isinstance(img, list))
    print('tail joint original')
    print(joints[-2:])

    print('head joint shuffled')
    print(joint_shuffled[:2])
    print('random state seed ', param.get('dataset_split_seed'))
    print('------')
    print('proprio unshuffled MEAN')
    check_mean(joints[index_from:])
    print('proprio shuffled MEAN')
    check_mean(joint_shuffled)
    print('------')
    print('proprio unshuffled STD')
    check_std(joints[index_from:])
    print('proprio shuffled STD')
    check_std(joint_shuffled)
    print('------')
    print('------')
    print('------')
    print('------')
    print('motor unshuffled MEAN')
    check_mean(cmd[index_from:])
    print('motor shuffled MEAN')
    check_mean(cmd_shuffled)
    print('------')
    print('motor unshuffled STD')
    check_std(cmd[index_from:])
    print('motor shuffled STD')
    check_std(cmd_shuffled)