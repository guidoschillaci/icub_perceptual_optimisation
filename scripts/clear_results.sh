echo "Deleting Optical Flow dataset"
rm datasets/icub_alone/dataset_optical_flow.npy
rm datasets/only_ball/dataset_optical_flow.npy
rm datasets/icub_and_ball/dataset_optical_flow.npy
rm datasets/icub_and_many_balls/dataset_optical_flow.npy
rm datasets/combined_alone_and_many_balls/dataset_optical_flow.npy

#read -p "Do you want to remove the experiments folder? " -n 1 -r
#echo    # (optional) move to a new line
#if [[ $REPLY =~ ^[Yy]$ ]]
#then
    echo "Deleting old results and models"
    rm -r experiments/
    echo "removed!"
#fi
