# remember to install pyqt5 to show images with matplotlib, e.g.  pip3 install PyQt5==5.9.2

import yarp
import cv2
import time
import sys
import numpy as np

import matplotlib.pyplot as plt
import random

print(sys.argv)

hold_hands = False

class ExplorationModule(yarp.RFModule):
    def configure(self, rf):

        self.config = yarp.Property()
        self.config.fromConfigFile('/code/icub_perceptual_optimisation/yarp/config.ini')
        self.width = self.config.findGroup('CAMERA').find('width').asInt32()
        self.height = self.config.findGroup('CAMERA').find('height').asInt32()

        if self.config.findGroup('GENERAL').find('show_images').asBool():
            import matplotlib
            matplotlib.use('TKAgg')
            self.ax_left = plt.subplot(1, 2, 1)
            self.ax_right = plt.subplot(1, 2, 2)

        # prepare motor driver
        self.head_motors = 'head'
        self.head_motorprops = yarp.Property()
        self.head_motorprops.put("device", "remote_controlboard")
        self.head_motorprops.put("local", "/client_babbling/" + self.head_motors)
        self.head_motorprops.put("remote", "/icubSim/" + self.head_motors)

        self.left_motors = 'left_arm'
        self.left_motorprops = yarp.Property()
        self.left_motorprops.put("device", "remote_controlboard")
        self.left_motorprops.put("local", "/client_babbling/" + self.left_motors)
        self.left_motorprops.put("remote", "/icubSim/" + self.left_motors)

        self.right_motors = 'right_arm'
        self.right_motorprops = yarp.Property()
        self.right_motorprops.put("device", "remote_controlboard")
        self.right_motorprops.put("local", "/client_babbling/" + self.right_motors)
        self.right_motorprops.put("remote", "/icubSim/" + self.right_motors)

        # create remote driver
        self.head_driver = yarp.PolyDriver(self.head_motorprops)
        print('head motor driver prepared')
        # query motor control interfaces
        self.head_iPos = self.head_driver.viewIPositionControl()
        self.head_iVel = self.head_driver.viewIVelocityControl()
        self.head_iEnc = self.head_driver.viewIEncoders()
        self.head_iCtlLim = self.head_driver.viewIControlLimits()

        self.left_armDriver = yarp.PolyDriver(self.left_motorprops)
        print('left motor driver prepared')
        # query motor control interfaces
        self.left_iPos = self.left_armDriver.viewIPositionControl()
        self.left_iVel = self.left_armDriver.viewIVelocityControl()
        self.left_iEnc = self.left_armDriver.viewIEncoders()
        self.left_iCtlLim = self.left_armDriver.viewIControlLimits()

        self.right_armDriver = yarp.PolyDriver(self.right_motorprops)
        print('right motor driver prepared')
        # query motor control interfaces
        self.right_iPos = self.right_armDriver.viewIPositionControl()
        self.right_iVel = self.right_armDriver.viewIVelocityControl()
        self.right_iEnc = self.right_armDriver.viewIEncoders()
        self.right_iCtlLim = self.right_armDriver.viewIControlLimits()

        #  number of joints
        self.num_joints = self.left_iPos.getAxes()
        print('Num joints: ', self.num_joints)

        self.head_num_joints = self.head_iPos.getAxes()
        print('Num head joints: ', self.head_num_joints)

        self.head_limits = []
        for i in range(self.head_num_joints):
            head_min =yarp.DVector(1)
            head_max =yarp.DVector(1)
            self.head_iCtlLim.getLimits(i, head_min, head_max)
            print('lim head ', i, ' ', head_min[0], ' ', head_max[0])
            self.head_limits.append([head_min[0], head_max[0]])

        self.left_limits = []
        self.right_limits = []
        for i in range(self.num_joints):
            left_min =yarp.DVector(1)
            left_max =yarp.DVector(1)
            self.left_iCtlLim.getLimits(i, left_min, left_max)
            #print('lim left ', i, ' ', left_min[0], ' ', left_max[0])
            self.left_limits.append([left_min[0], left_max[0]])

            right_min =yarp.DVector(1)
            right_max =yarp.DVector(1)
            self.right_iCtlLim.getLimits(i, right_min, right_max)
            #print('lim right ', i, ' ', right_min[0], ' ', right_max[0])
            self.right_limits.append([right_min[0], right_max[0]])

        self.go_to_starting_pos()

        moduleName = rf.check("name", yarp.Value("BabblingModule")).asString()
        self.setName(moduleName)
        print('module name: ',moduleName)
        yarp.delay(5.0)
        print('starting now')

    def close(self):
        print("Going to starting position and closing")
        #self.go_to_starting_pos()

    def interruptModule(self):
        return True

    def getPeriod(self):
        return 6 # seconds

    def babble_arm(self):

        if not self.left_iPos.checkMotionDone() and not self.right_iPos.checkMotionDone():
            print ('waiting for movement to finish...')
        else:
            print ('new movement')
        target_left_pos =  self.left_startingPos
        target_right_pos =  self.right_startingPos
        for i in (*range(0,4), *range(7,16)):
            if self.config.findGroup('GENERAL').find('babble_left').asBool():
                target_left_pos[i] = random.uniform(self.left_limits[i][0], self.left_limits[i][1])
            if self.config.findGroup('GENERAL').find('babble_right').asBool():
                target_right_pos[i] = random.uniform(self.right_limits[i][0], self.right_limits[i][1])

        print ('sending command left ', target_left_pos.toString())
        self.left_iPos.positionMove(target_left_pos.data())
        print ('sending command right ', target_right_pos.toString())
        self.right_iPos.positionMove(target_right_pos.data())

        return target_left_pos, target_right_pos

    def go_to_starting_pos(self):
        start_head = yarp.Vector(self.head_num_joints)
        start_head[0] = -25
        start_head[1] = 0
        start_head[2] = 40
        start_head[3] = 0
        start_head[4] = 0
        start_head[5] = 0

        # starting position (open hand in front on the left camera
        start_left = yarp.Vector(self.num_joints)
        start_left[0] = -80 # l_shoulder_pitch
        start_left[1] = 16 # l_shoulder_roll
        start_left[2] = 30 # l_shoulder_yaw
        start_left[3] = 65 # l_elbow
        start_left[4] = -5 # was -80 # l_wrist_prosup
        start_left[5] = 0 # l_wrist_pitch
        start_left[6] = 0 # l_wrist_yaw
        start_left[7] = 58.8 # l_hand_finger adduction/abduction
        start_left[8] = 20 # l_thumb_oppose
        start_left[9] = 19.8 # l_thumb_proximal
        start_left[10] = 19.8 # l_thumb_distal
        start_left[11] = 9.9 # l_index_proximal
        start_left[12] = 10.8 # l_index_distal
        start_left[13] = 9.9 # l_middle_proximal
        start_left[14] = 10.8 # l_middle_distal
        start_left[15] = 10.8 # l_pinky

        start_right = yarp.Vector(self.num_joints)
        start_right[0] = -40
        start_right[1] = 16
        start_right[2] = 70
        start_right[3] = 80
        start_right[4] = -5 # was -80
        start_right[5] = 0
        start_right[6] = 0
        start_right[7] = 58.8
        start_right[8] = 20
        start_right[9] = 19.8
        start_right[10] = 19.8
        start_right[11] = 9.9
        start_right[12] = 10.8
        start_right[13] = 9.9
        start_right[14] = 10.8
        start_right[15] = 10.8

        self.head_startingPos = yarp.Vector(self.head_num_joints, start_head.data())
        self.left_startingPos = yarp.Vector(self.num_joints, start_left.data())
        self.right_startingPos = yarp.Vector(self.num_joints, start_right.data())
        self.head_iPos.positionMove(self.head_startingPos.data())
        if not hold_hands:
            self.left_iPos.positionMove(self.left_startingPos.data())
            self.right_iPos.positionMove(self.right_startingPos.data())

    # main function, called periodically every getPeriod() seconds
    def updateModule(self):
        #self.read_image()
        #self.read_skin()
        if not hold_hands:
            self.babble_arm()
        return True

yarp.Network.init()


rf = yarp.ResourceFinder()
rf.setVerbose(True);
rf.setDefaultContext("testContext");
rf.setDefaultConfigFile("default.ini");
rf.configure(sys.argv)

mod = ExplorationModule()
mod.configure(rf)

# check if run calls already configure(rf). upon success, the module execution begins with a call to updateModule()
mod.runModule()
