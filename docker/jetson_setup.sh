# first steps:
# git clone https://github.com/guidoschillaci/icub_perceptual_optimisation.git
# cd icub_perceptual_optimisation
# sudo usermod -aG docker jetson
# log out and in
# sudo apt-get update -y
# sudo apt install -y docker docker.io

wget https://zenodo.org/record/4593079/files/datasets.tar.xz --no-check-certificate
tar xvf datasets.tar.xz
rm datasets.tar.xz