# first steps:
# git clone https://github.com/guidoschillaci/icub_perceptual_optimisation.git
# cd icub_sensory_enhancement
# sudo usermod -aG docker cloudsigma
# log out and in
git pull
sudo apt-get update -y
sudo apt-get install -y \
    apt-transport-https \
    ca-certificates \
    curl \
    gnupg-agent \
    software-properties-common

curl -fsSL https://download.docker.com/linux/ubuntu/gpg | sudo apt-key add -

sudo add-apt-repository \
   "deb [arch=amd64] https://download.docker.com/linux/ubuntu \
   $(lsb_release -cs) \
   stable"
sudo apt-get update -y
sudo apt-get install -y docker-ce docker-ce-cli containerd.io


# install nvidia drivers and cuda
sudo apt install -y linux-headers-$(uname -r)
sudo curl -O https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2004/x86_64/cuda-ubuntu2004.pin
sudo mv cuda-ubuntu2004.pin /etc/apt/preferences.d/cuda-repository-pin-600
sudo apt-key adv --fetch-keys https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2004/x86_64/7fa2af80.pub
sudo add-apt-repository "deb https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2004/x86_64/ /"
sudo apt update
sudo apt -y install cuda

distribution=$(. /etc/os-release;echo $ID$VERSION_ID) \
   && curl -s -L https://nvidia.github.io/nvidia-docker/gpgkey | sudo apt-key add - \
   && curl -s -L https://nvidia.github.io/nvidia-docker/$distribution/nvidia-docker.list | sudo tee /etc/apt/sources.list.d/nvidia-docker.list
sudo apt-get update
sudo apt-get install -y nvidia-docker2
sudo systemctl restart docker

wget https://zenodo.org/record/4660600/files/Predictive_Robots_Perceptual_Optimisation_Datasets.tar.xz --no-check-certificate
tar xvf Predictive_Robots_Perceptual_Optimisation_Datasets.tar.xz
rm Predictive_Robots_Perceptual_Optimisation_Datasets.tar.xz
#wget https://zenodo.org/record/4139790/files/dataset_images_grayscale.npy --no-check-certificate
#wget https://zenodo.org/record/4139790/files/dataset_joint_encoders.npy --no-check-certificate
#wget https://zenodo.org/record/4139790/files/dataset_motor_commands.npy --no-check-certificate
#wget https://zenodo.org/record/4139790/files/dataset_skin_values.npy --no-check-certificate
#wget https://zenodo.org/record/4139790/files/dataset_timestamps.npy --no-check-certificate

sudo groupadd docker
sudo usermod -aG docker ${USER}

echo "Testing nvidia-smi - if it does not run, check nvdia driver and cuda installations!"
nvidia-smi
# log out and log in
#su -s ${USER}
exit
#cd ..

#docker build -f docker/Dockerfile.deeplearn-tf2-cloud-gpu . -t guidoski/deeplearn:tf2-cloud-gpu
