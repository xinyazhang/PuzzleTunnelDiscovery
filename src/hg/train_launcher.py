"""
TRAIN LAUNCHER

"""

import six.moves.configparser as configparser
from hourglass_tiny import HourglassModel
import datagen
import sys
DataGenerator = datagen.DataGenerator

def process_config(conf_file):
    """
    """
    params = {}
    config = configparser.ConfigParser()
    config.read(conf_file)
    for section in config.sections():
        if section == 'DataSetHG':
            for option in config.options(section):
                params[option] = eval(config.get(section, option))
        if section == 'Network':
            for option in config.options(section):
                params[option] = eval(config.get(section, option))
        if section == 'Train':
            for option in config.options(section):
                params[option] = eval(config.get(section, option))
        if section == 'Validation':
            for option in config.options(section):
                params[option] = eval(config.get(section, option))
        if section == 'Saver':
            for option in config.options(section):
                params[option] = eval(config.get(section, option))
    return params

def craft_dict(params):
    dic = {}
    for k in ['suppress_hot', 'red_noise', 'suppress_cold']:
        if k in params:
            dic[k] = params[k]
        else:
            dic[k] = 0.0
    return dic

if __name__ == '__main__':
    cfile = 'config.cfg' if len(sys.argv) < 2 else sys.argv[1]
    print('--Parsing Config File {}'.format(cfile))
    params = process_config(cfile)

    print('--Creating Dataset')
    if 'new_dataset' not in params:
        dataset = DataGenerator(params['joint_list'], params['img_directory'], params['training_txt_file'], remove_joints=params['remove_joints'])
        dataset._create_train_table()
        dataset._randomize()
        dataset._create_sets()
        ds_name = ''
    else:
        ds_name = params['new_dataset']
        aug_dict = craft_dict(params)
        dataset = datagen.create_dataset(ds_name, aug_patch=True, aug_scaling=0.5, aug_dict=aug_dict)
        params['num_joints'] = dataset.d_dim
        assert params['weighted_loss'] is False, "No support for weighted loss for now"

    is_testing = False if 'do_testing' not in params else params['do_testing']

    model = HourglassModel(nFeat=params['nfeats'],
                               nStack=params['nstacks'],
                               nModules=params['nmodules'],
                               nLow=params['nlow'],
                               outputDim=params['num_joints'],
                               batch_size=params['batch_size'],
                               attention=params['mcam'],
                               training=not is_testing,
                               drop_rate=params['dropout_rate'],
                               lear_rate=params['learning_rate'],
                               decay=params['learning_rate_decay'],
                               decay_step=params['decay_step'],
                               dataset=dataset,
                               dataset_name=ds_name,
                               name=params['name'],
                               logdir_train=params['log_dir_train'],
                               logdir_test=params['log_dir_test'],
                               tiny=params['tiny'],
                               w_loss=params['weighted_loss'],
                               joints= params['joint_list'],
                               modif=False)
    model.generate_model()
    if is_testing:
        model.testing_init(nEpochs=1, epochSize=params['epoch_size'], saveStep=0, dataset=None, load=params['name'])
    else:
        model.training_init(nEpochs=params['nepochs'], epochSize=params['epoch_size'], saveStep=params['saver_step'], dataset = None)
        # model.training_init(nEpochs=params['nepochs'], epochSize=params['epoch_size'], saveStep=params['saver_step'], dataset = None, load=params['name'])

