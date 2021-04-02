# readapted from https://github.com/robotology/icub-tutorials/blob/master/python/python_simworld_control.py

import collections
import yarp
import time
import random

yarp.Network.init() # Initialise YARP
simulate_balls = True
many_balls = True

class WorldController:
    """Class for controlling iCub simulator via its RPC world port."""

    def __init__(self):
        self._rpc_client = yarp.RpcClient()
        self._port_name = "/WorldController-" + str(id(self)) + "/commands"
        self._rpc_client.open(self._port_name)
        self._rpc_client.addOutput("/icubSim/world")

        # A dictionary to track simulator object IDs for all types of objects
        self._sim_ids_counters = collections.defaultdict(lambda: 0)

        # A sequence to track internal object IDs. This list stores tuples (object type, simulator id)
        # so that outside one does not have to remember the type of object.
        self._objects = []

        # A sequence to track markers
        self._markers = []

    def _execute(self, cmd):
        """Execute an RPC command, returning obtained answer bottle."""
        ans = yarp.Bottle()
        self._rpc_client.write(cmd, ans)
        return ans
    def _is_success(self, ans):
        """Check if RPC call answer Bottle indicates successfull execution."""
        return ans.size() == 1 and ans.get(0).asVocab() == 27503 # Vocab for '[ok]'


    def _prepare_del_all_command(self):
        """Prepare the "world del all" command bottle."""
        result = yarp.Bottle()
        result.clear()
        #result.addString(["world", "del", "all"])
        result.addString("world")
        result.addString("del")
        result.addString("all")
        return result


    def del_all(self):
        """Delete all objects from the simultor"""
        result = self._is_success(self._execute(self._prepare_del_all_command()))

        if result:
            # Clear the counters
            self._sim_ids_counters.clear()
            del self._objects[:]
            del self._markers[:]

        return result


    def _prepare_create_command(self, obj, size, location, colour):
        """Prepare an RPC command for creating an object in the simulator environment.

        See Simulator Readme section 'Object Creation'

        Parameters:
            obj - object type string. 'sph', 'box', 'cyl' 'ssph', 'sbox' or 'scyl'.
            size - list of values specifying the size of an object. Parameters depend on object type:
                (s)box: [ x, y, z ]
                (s)sph: [ radius ]
                (s)cyl: [ radius, length ]
            location - coordinates of the object location, [ x, y, z ]
            colour - object colour in RGB (normalised), [ r, g, b ]
        Returns:
            yarp.Bottle with the command, ready to be sent to the rpc port of the simulator

        """

        result = yarp.Bottle()
        result.clear()
        #result.addString(["world", "mk", obj])
        result.addString("world")
        result.addString("mk")
        result.addString(obj)
        for i in range(len(size)):
            result.addDouble(size[i])
        for i in range(len(location)):
            result.addDouble(location[i])
        for i in range(len(colour)):
            result.addDouble(colour[i])
        return result

    def create_object(self, obj, size, location, colour):
        """Create an object of a specified type, size, location and colour, returning internal object ID or -1 on error."""
        print (obj)
        cmd = self._prepare_create_command(obj, size, location, colour)
        #print (cmd.toString())
        if self._is_success(self._execute(cmd)):
            obj_sim_id = self._sim_ids_counters[obj] + 1  # iCub simulator IDs start from 1

            # Update the counters
            self._sim_ids_counters[obj] += 1
            self._objects.append((obj, obj_sim_id))

            print ('object created')
            # Internal object IDs are shared among all types of objects and start from 0;
            # they are essentially indices of the self._objects sequence
            return len(self._objects) - 1
        else:
            print ('error in creating object')
            return -1  # error

    def create_markers(self, mark_id, location):
        """Create an Aruco marker of a specified type, size, location, returning internal object ID or -1 on error."""
        print ('marker id', mark_id)
        cmd = yarp.Bottle()
        cmd.clear()
        #result.addString(["world", "mk", obj])
        cmd.addString("world")
        cmd.addString("mk")
        cmd.addString("smodel")
        cmd.addString("marker_"+str(mark_id)+".x")
        #cmd.addString("marker.x")
        #cmd.addString("icosphere.x")
        #cmd.addString("icosphere.bmp")
        cmd.addString("marker_"+str(mark_id)+".bmp")
        for i in range(len(location)):
            cmd.addDouble(location[i])
        #for i in range(len(location)):
        #    cmd.addDouble(location[i])

        print (cmd.toString())
        if self._is_success(self._execute(cmd)):
            print ('marker created')
        else:
            print ('error in creating marker')
        rotate_cmd = yarp.Bottle()
        rotate_cmd.clear()
        rotate_cmd.addString("world")
        rotate_cmd.addString("rot")
        rotate_cmd.addString("smodel")
        rotate_cmd.addInt(mark_id)
        rotate_cmd.addDouble(0)
        rotate_cmd.addDouble(-65)
        rotate_cmd.addDouble(0)
        print (rotate_cmd.toString())
        if self._is_success(self._execute(rotate_cmd)):
            print ('rotated marker ', str(mark_id))


    def _prepare_move_command(self, obj, obj_id, location):
        """Prepare the "world set <obj> <xyz>" command bottle."""
        result = yarp.Bottle()
        result.clear()
        #result.addString(["world", "set", obj])
        result.addString("world")
        result.addString("set")
        result.addString(obj)
        result.addInt(obj_id)
        for i in range(len(location)):
            result.addDouble(location[i])
        return result

    def move_object(self, obj_id, location):
        """Move an object specified by the internal id to another location."""
        obj_desc = self._objects[obj_id]
        return self._is_success(self._execute(self._prepare_move_command(obj_desc[0], obj_desc[1], location)))

    def _prepare_get_command(self, obj, obj_id):
        """Prepare the "world get <obj> <id>" command bottle."""
        result = yarp.Bottle()
        result.clear()
        #result.addString(["world", "get", obj])
        result.addString("world")
        result.addString("get")
        result.addString(obj)
        result.addInt(obj_id)

        return result

    def get_object_location(self, obj_id):
        """Obtain the object location from the simulator. Returns None on failure."""
        obj_desc = self._objects[obj_id]
        result = self._execute(self._prepare_get_command(obj_desc[0], obj_desc[1]))
        if result.size() == 3:
            return [result.get(i).asDouble() for i in range(3)]  # 3-element list with xyz coordinates
        else:
            return None  # An error occured

    def __del__(self):
        try:
            if self._rpc_client is not None:
                self.del_all()
            self._rpc_client.close()
            del self._rpc_client
        except AttributeError:
            pass

    def rotate(self, mark_id, rotation):
        rotate_cmd = yarp.Bottle()
        rotate_cmd.clear()
        rotate_cmd.addString("world")
        rotate_cmd.addString("rot")
        rotate_cmd.addString("smodel")
        rotate_cmd.addInt(mark_id+1)
        rotate_cmd.addDouble(rotation[0])
        rotate_cmd.addDouble(rotation[1])
        rotate_cmd.addDouble(rotation[2])
        print (rotate_cmd.toString())
        if self._is_success(self._execute(rotate_cmd)):
            print ('rotated marker ', str(mark_id))
	



wc = WorldController()
config = yarp.Property()
config.fromConfigFile('/code/icub_perceptual_optimisation/yarp/config.ini')
max_num_objects = config.findGroup('GENERAL').find('max_num_objects').asInt32()
if many_balls:
    max_num_objects = max_num_objects * 3


# make marker objects
cmd = yarp.Bottle()
cmd.clear()
cmd.addString("world")
cmd.addString("set")
cmd.addString("mdir")
cmd.addString("/code/icub_perceptual_optimisation/yarp/data/markers")
if wc._is_success(wc._execute(cmd)):
    print ('changed model folder')
# create marker objects


wc.create_markers(0, [1.4, 0.15, 0.5 ])
wc.create_markers(1,  [0.85, 0.15, 0.75 ])
#wc.rotate(0,  [0, -35, -40 ])
#wc.rotate(1,  [0, 90, 0 ])
wc.create_markers(2, [0.4, 0.15, 1 ])
#wc.rotate(2,  [0, -55, -40 ])
wc.create_markers(3, [1.5, 0.55, 0.7 ])
#wc.rotate(3,  [0, -55, -40 ])
wc.create_markers(4, [0.9, 0.55, 0.85 ])
#wc.rotate(4,  [0, -55, -40 ])
wc.create_markers(5, [0.45, 0.5, 1.2 ])
#wc.rotate(5,  [0, -55, -40 ])
wc.create_markers(6, [1.5, 0.95, 0.85 ])
#wc.rotate(6,  [0, -55, -40 ])
wc.create_markers(7, [1, 0.9, 1.15 ])
#wc.rotate(7,  [0, -55, -20 ])
wc.create_markers(8, [0.5, 0.8, 1.4 ])
wc.rotate(8,  [0, -45, 0 ])



# create falling ball objects
while True:
    if simulate_balls:
        y = 1.0
        x = random.uniform(0, 0.3)
        z = random.uniform(0, 0.3)
        if len(wc._objects) >= max_num_objects:
            id = random.randint(0, max_num_objects-1)
            print('moving ball ',id)
            #del wc._objects[random.randint(0,9)]
            wc.move_object(id, [ x, y, z ])
        else:
            print ('creating ball')
            red_sphere = wc.create_object('sph', [ 0.03 ], [ x, y, z ], [ random.random(), random.random(), random.random() ]) # size, location, color
        if many_balls:
            time.sleep(2)
        else:
            time.sleep(20)

#green_box = wc.create_object('sbox', [ 0.1, 0.1, 0.1 ], [ 0, 0.1, 1 ], [ 0, 1, 0 ])
#blue_cylinder = wc.create_object('scyl', [ 0.05, 0.05 ], [ 1, 0.1, 0.1 ], [ 0, 0, 1 ])
# ... move them around ...
#wc.move_object(red_sphere, [ 1, 1, 0 ])
#wc.move_object(green_box, [ -1, 0.5, 1 ])
#wc.move_object(blue_cylinder, [ 0, 0.5, 1 ])
# ... and ask for their positions ...
#print (wc.get_object_location(green_box))
# Now let's cleanup
#wc.del_all()
#  delete the wc object and invoke its destructor
#del wc

time.sleep(5)
