xhost +local:root

ipclient=''

if [ -z "$1" ]
  then
    echo "No argument supplied. I set the address of the client pc to ${ipclient}."
  else
    ipclient=$1
    echo "Setting the address of the client pc to ${ipclient}"
fi

DISPLAY=$ipclient:0

export DOCKER_CONTAINER_NAME=icub_container
if [ ! "$(docker ps -q -f name=${DOCKER_CONTAINER_NAME})" ]; then
    if [ "$(docker ps -aq -f status=exited -f name=${DOCKER_CONTAINER_NAME})" ]; then
        echo "Cleaning up existing container named ${DOCKER_CONTAINER_NAME}"
        # cleanup
        docker rm $DOCKER_CONTAINER_NAME
    fi
    echo "Connecting to new container named ${DOCKER_CONTAINER_NAME}"
    # run your container 
    docker run -it --rm --privileged \
      -e DISPLAY=$DISPLAY \
      --name "$DOCKER_CONTAINER_NAME" \
      --volume="/tmp/.X11-unix:/tmp/.X11-unix:rw" \
      --volume="/home/guido/code:/code/:rw"  \
      --gpus all  \
      -w /code \
      guidoski/icub:tf2-nogpu bash -c 'bash'
else
    echo "Connecting to existing container named ${DOCKER_CONTAINER_NAME}"
    # run your container 
    docker exec -it --privileged \
      -e DISPLAY=$DISPLAY \
      "$DOCKER_CONTAINER_NAME" \
      bash  -c 'bash'
fi

#-e DISPLAY=docker.for.mac.host.internal:0 \


