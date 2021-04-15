echo "Deleting Optical Flow dataset"
rm
#read -p "Do you want to remove the experiments folder? " -n 1 -r
#echo    # (optional) move to a new line
#if [[ $REPLY =~ ^[Yy]$ ]]
#then
    echo "Deleting old results and models"
    rm -r experiments/
    echo "removed!"
#fi
