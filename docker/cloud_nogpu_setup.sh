# first steps:
# git clone https://github.com/guidoschillaci/icub_perceptual_optimisation.git
# cd icub_sensory_enhancement
# sudo usermod -aG docker cloudsigma
# log out and in
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
sudo apt-get install docker-ce docker-ce-cli containerd.io

#mkdir results
#mkdir results/plots
#mkdir results/plots/gif
#mkdir results/models
#mkdir datasets
#cd datasets
wget https://zenodo.org/record/4593079/files/datasets.tar.xz --no-check-certificate
tar xzf datasets.tar.xz
rm datasets.tar.xz
#wget https://zenodo.org/record/4139790/files/dataset_images_grayscale.npy --no-check-certificate
#wget https://zenodo.org/record/4139790/files/dataset_joint_encoders.npy --no-check-certificate
#wget https://zenodo.org/record/4139790/files/dataset_motor_commands.npy --no-check-certificate
#wget https://zenodo.org/record/4139790/files/dataset_skin_values.npy --no-check-certificate
#wget https://zenodo.org/record/4139790/files/dataset_timestamps.npy --no-check-certificate



#cd ..

# docker build -f docker/Dockerfile.deeplearn-tf2-nogpu . -t guidoski/deeplearn:tf2-nogpu
