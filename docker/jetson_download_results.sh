echo "Clearing previous results"
sh docker/jetson_clear_results.sh
echo "Downloading from jetson"
scp -r jetson@192.168.1.155:/home/jetson/code/icub_perceptual_optimisation/results ./
echo "Results and plots downloaded!"