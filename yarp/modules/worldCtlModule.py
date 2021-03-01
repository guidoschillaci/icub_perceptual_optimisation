# readapted from https://github.com/robotology/icub-tutorials/blob/master/python/python_simworld_control.py

import collections
import yarp
import time
import random

yarp.Network.init() # Initialise YARP


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





wc = WorldController()
config = yarp.Property()
config.fromConfigFile('/code/icub_intrinsic_motivation/yarp/config.ini')
max_num_objects = config.findGroup('GENERAL').find('max_num_objects').asInt32()

# create marker objects
marker_obj_0 = wc.create_object('sbox', [ 0.1, 0.1, 0.1 ], [ 0, 0.1, 1 ], [ 0, 1, 0 ])
marker_obj_1 = wc.create_object('sbox', [ 0.1, 0.1, 0.1 ], [ 0, 0.1, 1 ], [ 0, 1, 0 ])
marker_obj_2 = wc.create_object('sbox', [ 0.1, 0.1, 0.1 ], [ 0, 0.1, 1 ], [ 0, 1, 0 ])
marker_obj_3 = wc.create_object('sbox', [ 0.1, 0.1, 0.1 ], [ 0, 0.1, 1 ], [ 0, 1, 0 ])
marker_obj_4 = wc.create_object('sbox', [ 0.1, 0.1, 0.1 ], [ 0, 0.1, 1 ], [ 0, 1, 0 ])
marker_obj_5 = wc.create_object('sbox', [ 0.1, 0.1, 0.1 ], [ 0, 0.1, 1 ], [ 0, 1, 0 ])
marker_obj_6 = wc.create_object('sbox', [ 0.1, 0.1, 0.1 ], [ 0, 0.1, 1 ], [ 0, 1, 0 ])
marker_obj_7 = wc.create_object('sbox', [ 0.1, 0.1, 0.1 ], [ 0, 0.1, 1 ], [ 0, 1, 0 ])
marker_obj_8 = wc.create_object('sbox', [ 0.1, 0.1, 0.1 ], [ 0, 0.1, 1 ], [ 0, 1, 0 ])

while True:
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