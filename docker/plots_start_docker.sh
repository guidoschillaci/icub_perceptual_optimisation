git pull

xhost +local:root

ipclient=localhost

if [ -z "$1" ]
  then
    echo "No argument supplied. I set the address of the client pc to localhost."
  else
    ipclient=$1
    echo "Setting the address of the client pc to ${ipclient}"
fi


export DOCKER_CONTAINER_NAME=deeplearn_plot_container
if [ ! "$(docker ps -q -f name=${DOCKER_CONTAINER_NAME})" ]; then
    if [ "$(docker ps -aq -f status=exited -f name=${DOCKER_CONTAINER_NAME})" ]; then
        echo "Cleaning up existing container named ${DOCKER_CONTAINER_NAME}"
        # cleanup
        docker rm $DOCKER_CONTAINER_NAME
    fi
    echo "Connecting to new container named ${DOCKER_CONTAINER_NAME}"
    # run your container 
    docker run -it --rm \
      -e DISPLAY=$DISPLAY \
      --name "$DOCKER_CONTAINER_NAME" \
      --volume="/Volumes/data/code/:/code/:rw"  \
      --volume="/tmp/.X11-unix:/tmp/.X11-unix:rw" \
      --workdir="/code" \
      guidoski/deeplearn:plots bash -c 'bash'
else
    echo "Connecting to existing container named ${DOCKER_CONTAINER_NAME}"
    # run your container 
    docker exec -it \
      -e DISPLAY=$DISPLAY \
      "$DOCKER_CONTAINER_NAME" \
      bash  -c 'bash'
fi

#-e DISPLAY=docker.for.mac.host.internal:0 \


