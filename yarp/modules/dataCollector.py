# author: Guido Schillaci, The BioRobotics Institute, Scuola Superiore Sant'Anna, Pisa, Italy
# email: guido.schillaci@santannapisa.it

import yarp
import cv2
import time
import sys
import numpy as np
from copy import deepcopy
import matplotlib.pyplot as plt
import random
import threading

class DataCollector(yarp.RFModule):
    def configure(self, rf):
        self.lock = threading.Lock()

        self.config = yarp.Property()
        self.config.fromConfigFile('/code/icub_perceptual_optimisation/yarp/config.ini')
        self.width = self.config.findGroup('CAMERA').find('width').asInt32()
        self.height = self.config.findGroup('CAMERA').find('height').asInt32()
        self.max_dataset_size = self.config.findGroup('GENERAL').find('max_dataset_size').asInt32()

        if self.config.findGroup('GENERAL').find('show_images').asBool():
            import matplotlib
            matplotlib.use('TKAgg')
            self.ax_left = plt.subplot(1, 2, 1)
            self.ax_right = plt.subplot(1, 2, 2)

        # Create a port and connect it to the iCub simulator virtual camera
        self.input_port_cam = yarp.Port()
        self.input_port_cam.open("/dataCollector/camera_left")
        yarp.Network.connect("/icubSim/cam/left", "/dataCollector/camera_left")

        self.input_port_skin_left_hand = yarp.Port()
        self.input_port_skin_left_hand.open("/dataCollector/skin/left_hand_comp") #
        yarp.Network.connect("/icubSim/skin/left_hand_comp", "/dataCollector/skin/left_hand_comp")

        self.input_port_skin_left_forearm = yarp.Port()
        self.input_port_skin_left_forearm.open("/dataCollector/skin/left_forearm_comp") #
        yarp.Network.connect("/icubSim/skin/left_forearm_comp", "/dataCollector/skin/left_forearm_comp")

        self.input_port_command = yarp.Port()
        self.input_port_command.open("/dataCollector/command") #
        yarp.Network.connect("/client_babbling/left_arm/command", "/dataCollector/command")

        #self.input_port_skin.read(False);    #  clean the buffer
        #self.input_port_skin.read(False);    #  clean the buffer

        # prepare image
        self.yarp_img_in = yarp.ImageRgb()
        self.yarp_img_in.resize(self.width, self.height)
        self.img_array = np.ones((self.height, self.width, 3), dtype=np.uint8)
        # yarp image will be available in self.img_array
        self.yarp_img_in.setExternal(self.img_array.data, self.width, self.height)

        # prepare motor driver
        self.head_motors = 'head'
        self.head_motorprops = yarp.Property()
        self.head_motorprops.put("device", "remote_controlboard")
        self.head_motorprops.put("local", "/client_datacollector/" + self.head_motors)
        self.head_motorprops.put("remote", "/icubSim/" + self.head_motors)

        self.left_motors = 'left_arm'
        self.left_motorprops = yarp.Property()
        self.left_motorprops.put("device", "remote_controlboard")
        self.left_motorprops.put("local", "/client_datacollector/" + self.left_motors)
        self.left_motorprops.put("remote", "/icubSim/" + self.left_motors)

        #self.right_motors = 'right_arm'
        #self.right_motorprops = yarp.Property()
        #self.right_motorprops.put("device", "remote_controlboard")
        #self.right_motorprops.put("local", "/client_datacollector/" + self.right_motors)
        #self.right_motorprops.put("remote", "/icubSim/" + self.right_motors)

        # create remote driver
        self.head_driver = yarp.PolyDriver(self.head_motorprops)
        self.head_iEnc = self.head_driver.viewIEncoders()
        self.head_iPos = self.head_driver.viewIPositionControl()


        self.left_armDriver = yarp.PolyDriver(self.left_motorprops)
        self.left_iEnc = self.left_armDriver.viewIEncoders()
        self.left_iPos = self.left_armDriver.viewIPositionControl()

        #self.right_armDriver = yarp.PolyDriver(self.right_motorprops)
        #self.right_iEnc = self.right_armDriver.viewIEncoders()
        #self.right_iPos = self.right_armDriver.viewIPositionControl()

        #  number of joints
        self.num_joints = self.left_iPos.getAxes()
        self.head_num_joints = self.head_iPos.getAxes()

        moduleName = rf.check("name", yarp.Value("DataCollectorModule")).asString()
        self.setName(moduleName)
        print('module name: ',moduleName)

        self.skin_left_hand_input = yarp.Bottle()
        self.skin_left_forearm_input = yarp.Bottle()
        #self.left_command_input = yarp.Bottle()
        self.left_command_input = yarp.Vector(self.num_joints)

        self.dataset_timestamps = []
        self.dataset_images = []
        self.dataset_skin_values = []
        self.dataset_joint_encoders = []
        self.dataset_motor_commands = []

        print('starting now')

    def close(self):
        print("Closing datacollector module")


    def saveDatasets(self):
        print('saving dataset...')
        min_len = np.min(np.asarray([len(self.dataset_timestamps), \
                                     len(self.dataset_images), \
                                     len(self.dataset_joint_encoders), \
                                     len(self.dataset_motor_commands), \
                                     len(self.dataset_skin_values)]))
        # print('min ', min_len)
        if len(self.dataset_timestamps) > min_len:
            self.dataset_timestamps.pop()
            # print('popped timestamp')
        if len(self.dataset_images) > min_len:
            self.dataset_images.pop()
            # print('popped dataset_images')
        if len(self.dataset_joint_encoders) > min_len:
            self.dataset_joint_encoders.pop()
            # print('popped dataset_joint_encoders')
        if len(self.dataset_motor_commands) > min_len:
            self.dataset_motor_commands.pop()
            # print('popped dataset_motor_commands')
        if len(self.dataset_skin_values) > min_len:
            self.dataset_skin_values.pop()
            # print('popped dataset_skin_values')

        print(len(self.dataset_timestamps), ' timestamps')
        print(len(self.dataset_images), ' images')
        print(len(self.dataset_joint_encoders), ' joint encoders')
        print(len(self.dataset_motor_commands), ' motor commands')
        print(len(self.dataset_skin_values), ' skin values')

        np.save('results/dataset_timestamps.npy', self.dataset_timestamps)
        np.save('results/dataset_images.npy', self.dataset_images)
        np.save('results/dataset_joint_encoders.npy', self.dataset_joint_encoders)
        np.save('results/dataset_motor_commands.npy', self.dataset_motor_commands)
        np.save('results/dataset_skin_values.npy', self.dataset_skin_values)

        print ('making grayscale dataset...')
        self.dataset_images_grayscale = []
        # make grayscale dataset
        for i in range(len(self.dataset_images)):
            self.dataset_images_grayscale.append(deepcopy(cv2.cvtColor(self.dataset_images[i], cv2.COLOR_BGR2GRAY)))
        np.save('results/dataset_images_grayscale.npy', self.dataset_images_grayscale)
        print ('dataset saved!')

    def interruptModule(self):
        self.saveDatasets()
        return True

    def getPeriod(self):
        return 0.2 # seconds

    def read_encoders(self):
        left_encs=yarp.Vector(self.num_joints)
        #right_encs=yarp.Vector(self.num_joints)
        self.left_iEnc.getEncoders(left_encs.data())
        #self.right_iEnc.getEncoders(right_encs.data())
        data = []
        for i in range (self.num_joints):
            data.append(left_encs.get(i))
        self.dataset_joint_encoders.append(deepcopy(data))

    def read_skin(self):
        skin_data = []
        self.input_port_skin_left_hand.read(self.skin_left_hand_input);
        #self.input_port_skin_left_forearm.read(self.skin_left_forearm_input);
        for i in range(self.skin_left_hand_input.size()):
            skin_data.append( deepcopy(self.skin_left_hand_input.get(i).asDouble()) )
        for i in range(self.skin_left_forearm_input.size()):
            skin_data.append(deepcopy(self.skin_left_forearm_input.get(i).asDouble()))
        self.dataset_skin_values.append(skin_data)

    def read_image(self):
        # read image
        self.input_port_cam.read(self.yarp_img_in)
        # scale down img_array and convert it to cv2 image
        self.image = cv2.resize(self.img_array, (64, 64), interpolation=cv2.INTER_LINEAR)
        # append to dataset
        self.dataset_images.append(self.image)

        if self.config.findGroup('GENERAL').find('show_images').asBool():
            # display the image that has been read
            self.ax_left.imshow(self.image)
            plt.pause(0.01)
            plt.ion()

    def read_commands(self):
        command = []
        #self.input_port_command.read(self.left_command_input)
        self.left_iPos.getTargetPositions( self.left_command_input.data())
        for i in range(self.num_joints):
            #print ('command ', i , ' ' ,self.left_command_input.get(i) )
            command.append(self.left_command_input.get(i))
        #for i in range(self.left_command_input.size()):
        #    print ('command ', i , ' ', self.left_command_input.get(i).asDouble())
        #    self.motor_command_dataset.append(deepcopy(self.left_command_input.get(i).asDouble()))

        self.dataset_motor_commands.append(deepcopy (command))

    def get_time_ms(self):
        return int(round(time.time() * 1000))

    # main function, called periodically every getPeriod() seconds
    def updateModule(self):
        self.lock.acquire()
        time_1 = self.get_time_ms()
        self.dataset_timestamps.append(time_1)
        print('Dataset size ', len(self.dataset_timestamps), ' of ', \
              str(self.config.findGroup('GENERAL').find('max_dataset_size').asInt32()), \
              ' time ', time_1, ' millisec')
        self.read_commands()
        self.read_encoders()
        self.read_image()
        self.read_skin()

        self.lock.release()

        if len(self.dataset_timestamps) % self.max_dataset_size == 0 :
            self.saveDatasets()
            return False

        #time_2 = self.get_time_ms()
        #print ('time: ', time_2, ' reading data took ', str(time_2-time_1), ' milliseconds')
        return True


yarp.Network.init()

rf = yarp.ResourceFinder()
rf.setVerbose(True);
rf.setDefaultContext("testContext");
rf.setDefaultConfigFile("default.ini");
rf.configure(sys.argv)

dc_mod = DataCollector()
dc_mod.configure(rf)
# check if run calls already configure(rf). upon success, the module execution begins with a call to updateModule()
dc_mod.runModule()
