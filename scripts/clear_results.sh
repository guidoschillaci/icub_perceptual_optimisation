echo "Deleting Optical Flow dataset"
rm datasets/icub_alone/dataset_optical_flow.npy
rm datasets/icub_and_ball/dataset_optical_flow.npy
echo "Deleting old results and models"
rm -r experiments/
echo "done!"