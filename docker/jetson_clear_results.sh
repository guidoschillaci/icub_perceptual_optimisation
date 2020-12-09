echo "Deleting Optical Flow dataset"
rm datasets/dataset_optical_flow.npy
echo "Deleting old results and models"
rm -r results
echo "Creating results directories"
mkdir results
mkdir results/plots
mkdir results/plots/gif
mkdir results/models
echo "done!"