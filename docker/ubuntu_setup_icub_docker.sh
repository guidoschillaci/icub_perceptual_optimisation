# first steps:
# git clone https://github.com/guidoschillaci/icub_sensory_enhancement.git
# cd icub_sensory_enhancement
# sudo usermod -aG docker cloudsigma
# log out and in
# sudo apt-get update -y
# sudo apt install -y docker docker.io

docker build -f docker/Dockerfile.icub-sim . -t guidoski/icub:tf2-nogpu
