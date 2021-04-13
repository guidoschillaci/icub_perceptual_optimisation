ipclient=94.156.203.185
userid=cloudsigma
if [ -z "$1" ]
  then
    echo "No argument supplied. I set the username of the cloud instance to cloudsigma."
  else
    userid=$1
    echo "Setting the address of the cloud instance to ${userid}"
fi

if [ -z "$2" ]
  then
    echo "No argument supplied. I set the address of the cloud instance to localhost."
  else
    ipclient=$2
    echo "Setting the address of the cloud instance to ${ipclient}"
fi


#echo "Clearing previous results"
#sh scripts/clear_results.sh
echo "Downloading from ${userid}, IP ${ipclient} "
scp -r ${userid}@${ipclient}:/home/${userid}/icub_perceptual_optimisation/experiments .
echo "Results and plots downloaded!"