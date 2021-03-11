# first steps:
# git clone https://github.com/guidoschillaci/icub_perceptual_optimisation.git
# cd icub_sensory_enhancement
# sudo usermod -aG docker cloudsigma
# log out and in
docker build -f docker/Dockerfile.plots . -t guidoski/deeplearn:plots
