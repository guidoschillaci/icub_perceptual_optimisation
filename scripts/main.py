from models import Models
from parameters import Parameters

if __name__ == "__main__":
    # create parameters object
    param = Parameters()
    param.set('directory_results', 'results/')
    param.set('directory_models', 'results/models/')
    param.set('directory_plots', 'results/plots/')
    param.set('directory_plots_gif', 'results/plots/gif/')
    param.set('directory_datasets', 'datasets/')
    mod = Models(param)
    # load dataset
    mod.read_data()
    # build deep nn (or load it)
    mod.make_model()
    # save a plot of the model
    mod.plot_model()
    # train the network
    mod.train_model()

    # save plots
    mod.save_plots()
    # save nn
    mod.save_model()