ipclient=94.156.202.162

if [ -z "$1" ]
  then
    echo "No argument supplied. I set the address of the cloud instance to localhost."
  else
    ipclient=$1
    echo "Setting the address of the cloud instance to ${ipclient}"
fi


echo "Clearing previous results"
sh scripts/clear_results.sh
echo "Downloading from jetson"
scp -r cloudsigma@${ipclient}:/home/cloudsigma/icub_perceptual_optimisation/experiments .
echo "Results and plots downloaded!"