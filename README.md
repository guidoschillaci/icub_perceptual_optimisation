# Perceptual optimisation in the icub
Perceptual optimisation, multi-modal integration with precision/fusion weights using deep neural networks.
Tested on the iCub simulator

Preliminary dataset available here:
https://zenodo.org/record/4139790#.X5husEJKgUs

# how to run on Nvidia Jetson Nano

## First-time setup
ssh into the jetson and run 

```
sh docker/jetson_setup_docker.sh
```

## Run

ssh into the jetson. 
Run a docker container typing:

```
sh docker/jetson_start_docker.sh
```

Then, run

```
sh docker/jetson_run.sh
```

# how to run on a cloud remote instance

clone this repo, enter the main folder and run:

```
sh docker/cloud_setup.sh
```

then

```
sh docker/cloud_deeplearn.sh
```

To run the experiment:


```
python script/main.py
```

# Docker icub sim

Build docker for icub-sim

```
docker build -f docker/Dockerfile.icub-sim . -t guidoski/icub:tf2-nogpu
```


# Run icub sim

Open a terminal and run

```
sh docker/ubuntu_icub_yarpserver.sh
```

Open a new terminal and run

```
sh docker/ubuntu_icub_yarpmanager.sh
```

Open a new terminal and run

```
sh docker/ubuntu_icub_sim.sh
```


Open a new terminal and run

```
sh docker/ubuntu_icub_bash.sh
```


To start babbling, inside the docker bash, enter the folder icub_perceptual_optimisation and run

```
python3 yarp/modules/exploration.py
```


To start recording, inside a new docker bash (run a new ubuntu_icub_bash script), enter the folder icub_perceptual_optimisation and run

```
python3 yarp/modules/dataCollector.py
```

If you want objects during simulation, run a new ubuntu_icub_bash script and then:

```
python3 yarp/modules/worldCtlModule.py
```

