# Perceptual optimisation in the icub
Perceptual optimisation, multi-modal integration with precision/fusion weights using deep neural networks.
Tested on the iCub simulator

Preliminary dataset available here:
https://zenodo.org/record/4139790#.X5husEJKgUs


# how to run in a cloud remote instance

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

