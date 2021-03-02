echo "Deleting Optical Flow dataset"
rm datasets/dataset_optical_flow.npy
echo "Deleting old results and models"
rm -r experiments/
echo "done!"