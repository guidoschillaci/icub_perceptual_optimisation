# first steps:
# git clone https://github.com/guidoschillaci/icub_sensory_enhancement.git
# cd icub_sensory_enhancement
# sudo usermod -aG docker cloudsigma
# log out and in
# sudo apt-get update -y
# sudo apt install -y docker docker.io


#mkdir results
#mkdir results/plots
#mkdir results/plots/gif
#mkdir results/models
#mkdir datasets
#cd datasets
#wget https://zenodo.org/record/4139790/files/dataset_images_grayscale.npy --no-check-certificate
#wget https://zenodo.org/record/4139790/files/dataset_joint_encoders.npy --no-check-certificate
#wget https://zenodo.org/record/4139790/files/dataset_motor_commands.npy --no-check-certificate
#wget https://zenodo.org/record/4139790/files/dataset_skin_values.npy --no-check-certificate
#wget https://zenodo.org/record/4139790/files/dataset_timestamps.npy --no-check-certificate

#cd ..

# docker build -f docker/Dockerfile.deeplearn-tf2-latest-nogpu . -t guidoski/deeplearn:tf2
docker build -f docker/Dockerfile.deeplearn-cloud . -t guidoski/deeplearn:cloud-gpu
