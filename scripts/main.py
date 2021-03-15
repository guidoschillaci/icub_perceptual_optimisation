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
    # 0: robot alone in the scene; 1: robot and balls falling from the sky
    doe = build.build_full_fact( \
        {'dataset_train_type': [0], 'dataset_test_type': [1]})
        #{'dataset_train_type': [0, 1, 2], 'dataset_test_type': [0, 1]})
    #    {'dataset_type': [0, 1], 'attenuation_test_dataset_type': [0, 1]})
    # add case:
    # train: icub holding a starting position and balls falling from the sky
    # test: icub babbling alone in the scene
    #doe.append({'dataset_train_type': 2, 'dataset_test_type': 0}, ignore_index=True)
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
                directory_results = run_folder
                directory_models = run_folder + 'models/'
                directory_plots = run_folder + 'plots/'
                directory_plots_gif = run_folder + 'plots/gif/'
                #os.makedirs(directory_results)
                os.makedirs(directory_models)
                os.makedirs(directory_plots)
                os.makedirs(directory_plots_gif)
                # create parameters object
                param = Parameters()
                param.set('directory_results', run_folder)
                param.set('directory_models', run_folder+'models/')
                param.set('directory_plots', run_folder+'plots/')
                param.set('directory_plots_gif', run_folder+'plots/gif/')
                param.set('directory_datasets', datasets_folder)
                dataset_type = doe.loc[exp, 'dataset_train_type']
                if dataset_type == 0:
                    param.set('dataset_train_type', 'icub_alone')
                    param.set('directory_datasets_train', datasets_folder + 'icub_alone/')
                elif dataset_type == 1:
                    param.set('dataset_train_type', 'icub_and_ball')
                    param.set('directory_datasets_train', datasets_folder + 'icub_and_ball/')
                else:
                    param.set('dataset_train_type', 'only_ball')
                    param.set('directory_datasets_train', datasets_folder + 'only_ball/')

                test_dataset_type = doe.loc[exp, 'dataset_test_type']
                if test_dataset_type == 0:
                    param.set('dataset_test_type', 'icub_alone')
                    param.set('directory_datasets_test', datasets_folder + 'icub_alone/')
                elif test_dataset_type == 1:
                    param.set('dataset_test_type', 'icub_and_ball')
                    param.set('directory_datasets_test', datasets_folder + 'icub_and_ball/')
                else:
                    param.set('dataset_test_type', 'only_ball')
                    param.set('directory_datasets_test', datasets_folder + 'only_ball/')
                # create model
                mod = Models(param)
                # load dataset
                mod.read_data()
                # build deep nn (or load it)
                mod.make_model()
                # save a plot of the model
                mod.plot_model()
                # print parameters
                mod.parameters.print()
                mod.parameters.save()
                # train the network
                mod.train_model()

                # save plots
                mod.save_plots()
                # save nn
                mod.save_model()
