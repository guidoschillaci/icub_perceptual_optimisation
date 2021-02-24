from models import Models
from parameters import Parameters
import os
from doepy import build # design of experiments


if __name__ == "__main__":

    do_experiments = True
    number_of_runs = 5
    main_path = os.getcwd()
    datasets_folder = main_path + '/' + 'datasets/'
    multiple_experiments_folder = main_path + '/' + 'experiments'
    if not os.path.exists(multiple_experiments_folder):
        os.makedirs(multiple_experiments_folder)

    os.chdir(multiple_experiments_folder)

    # which train and test dataset to use
    doe = build.build_full_fact(
        {'dataset_type': [0, 1], 'attenuation_test_dataset_type': [0, 1]}) # 0: robot alone in the scene; 1: robot and balls falling from the sky
    print(doe)

    if do_experiments:
        doe.to_csv(multiple_experiments_folder + '/doe.csv', index=True, header=True)
        # for each row in the design of the experiment table
        for exp in range(doe.shape[0]):
            exp_folder = multiple_experiments_folder + '/exp' + str(exp)
            if not os.path.exists(exp_folder):
                os.makedirs(exp_folder)
            # repeat it for number_of_runs times
            for run in range(number_of_runs):
                print('Starting experiment n.', str(exp))
                run_folder = exp_folder + '/run_' + str(run) + '/'
                print('Current folder: ', run_folder)
                if not os.path.exists(run_folder):
                    os.makedirs(run_folder)
                os.chdir(run_folder)
                directory_results = run_folder +'results/'
                directory_models = run_folder + 'results/models/'
                directory_plots = run_folder + 'results/plots/'
                directory_plots_gif = run_folder + 'results/plots/gif/'
                os.makedirs(directory_results)
                os.makedirs(directory_models)
                os.makedirs(directory_plots)
                os.makedirs(directory_plots_gif)
                # create parameters object
                param = Parameters()
                param.set('directory_results', run_folder+'results/')
                param.set('directory_models', run_folder+'results/models/')
                param.set('directory_plots', run_folder+'results/plots/')
                param.set('directory_plots_gif', run_folder+'results/plots/gif/')

                dataset_type = doe.loc[exp, 'dataset_type']
                if dataset_type == 0:
                    param.set('dataset_type', 'robot_alone')
                    param.set('directory_datasets', datasets_folder + '/robot_alone/')
                else:
                    param.set('dataset_type', 'robot_and_ball')
                    param.set('directory_datasets', datasets_folder + '/robot_and_ball/')

                attenuation_test_dataset_type = doe.loc[exp, 'attenuation_test_dataset_type']
                if attenuation_test_dataset_type == 0:
                    param.set('attenuation_test_dataset_type', 'robot_alone')
                else:
                    param.set('attenuation_test_dataset_type', 'robot_and_ball')

                '''
                # create model
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
                '''