"""
TRAIN LAUNCHER

"""

import argparse
from os.path import join,dirname,basename,isabs,isfile
import six.moves.configparser as configparser
import sys

from . import util
from . import hg_datagen as datagen

_CONFIG_TEMPLATE= \
'''
# Commented arguments were removed in our implementation
[DataSetHG]
# training_txt_file: 'dataset.txt'
# img_directory: 'F:/Cours/DHPE/mpii_human_pose_v1/images'
img_size: 256
hm_size: 64
# num_joints: 16
# remove_joints: None
joint_list = ['tunnel_constructing_surface']

# New arguments
## config file of OMPL puzzle.
## Note: only accepts absolute path for simplicity.
ompl_config: '/file/to/ompl/config'
what_to_render: 'rob' # 'rob' for robot, 'env' for environment

[Network]
# 'name' was obsoluted due to confusion
# name: 'dual-ntrs/hg_ver3-env_flat/'
checkpoint_dir: '/CHECK/POINT/DIRECTORY/'
# Note: do NOT change the default value here.
#       Its default value is set by _process_config
epoch_to_load: -1
multichannel: False
fp16: False

nFeats: 256
nStacks: 2 # 2 is good enough
nModules: 1
tiny: False
nLow: 4
dropout_rate: 0.2
mcam: False

# New segment that describes the image augmentation
[Augmentation]
enable_augmentation: True
suppress_hot: 0.1
red_noise: 0.1
suppress_cold: 0.1

[Train]
batch_size: 8
nEpochs: 75
epoch_size: 1000
learning_rate: 0.00025
learning_rate_decay: 0.96
decay_step: 3000
weighted_loss: False

[Validation]
# valid_iteration: 10

[Prediction]
prediction_epoch_size: 4096
debug_predction: False

[Saver]
# Obsoluted, we now save logs under subdirectories of ckpt_dir
# log_dir_train: './logs/hg3-dual_tiny_env_flat-train'
# log_dir_test: './logs/hg3-dual_tiny_env_flat-test'
saver_step: 500
saver_directory: ''
'''

def _process_config(config):
    '''
    Well this parser simply flatten out all sections ...
    '''
    params = {}
    for section in config.sections():
        for option in config.options(section):
                params[option] = eval(config.get(section, option))
    params['epoch_to_load'] = None
    return params

def process_config(conf_file):
    config = configparser.ConfigParser()
    config.read(conf_file)
    return _process_config(config)

'''
It is easier to create new configurations from the template (default)
'''
def create_default_config():
    config = configparser.ConfigParser()
    config.read_string(_CONFIG_TEMPLATE)
    return _process_config(config)

def create_config_from_profile(name):
    ret = create_default_config()
    if name == 'hg4':
        ret['nstacks'] = 4
    elif name == '256hg':
        ret['nlow'] = 6
        ret['batch_size'] = 4
    elif name == '256hg+normal':
        ret['nlow'] = 6
        ret['batch_size'] = 4
        ret['training_data_include_surface_normal'] = 1
    elif name == '256hg+normal-aug':
        ret['nlow'] = 6
        ret['batch_size'] = 4
        ret['training_data_include_surface_normal'] = 1
        ret['enable_augmentation'] = False
    else:
        print(f"Unknown profile {name}")
        raise NotImplementedError()
    util.log("[create_config_from_profile] {}".format(ret))
    return ret

def create_config_from_tagstring(tagstring):
    ret = create_default_config()
    tags = sorted(tagstring.split('.'))
    if '256hg' in tags:
        ret['nlow'] = 6
        ret['batch_size'] = 4
    if 'hg4' in tags:
        ret['nstacks'] = 4
    if '+normal' in tags:
        ret['training_data_include_surface_normal'] = 1
    if '+weight' in tags:
        ret['weighted_loss'] = True
    if '+tiny' in tags:
        ret['tiny'] = True
    if '+fp16' in tags:
        ret['fp16'] = True
    if '-aug' in tags:
        ret['enable_augmentation'] = False
    if 'feat128' in tags:
        ret['nfeats'] = 128
    if '+multichannel' in tags:
        # Note 'joints' should also be modified, but we do not have this info yet.
        #      This info is added at hg_datagen.create_dataset_from_params()
        #      Also, the 'weighted_loss' should be orthogonal to mutichannel; they share the same loss function,
        #      but with different way to generate weigths.
        ret['multichannel'] = True
    if 'lowmem' in tags:
        ret['batch_size'] = 2
        ret['epoch_size'] = 2000
    util.log("[create_config_from_tagstring] {} -> {} -> {}".format(tagstring, tags, ret))
    return ret, '.'.join(tags)

def launch_with_params(params, do_training, load=False):
    print('--Creating Dataset')
    # According to workspace hierarchy, the foloder name is the actual puzzle name
    # Note: all traing data are named after 'train'
    assert 'dataset_name' in params
    ds_name = params['dataset_name']
    dataset = datagen.create_dataset_from_params(params)
    params['nepochs'] = 25 + 75 * dataset.number_of_geometries
    if 'max_epoch' in params:
        params['nepochs'] = params['max_epoch']

    assert dataset.d_dim == 4, f'{dataset.d_dim} != 4'
    params['num_joints'] = dataset.d_dim
    assert params['weighted_loss'] is False, "No support for weighted loss for now"

    util.log("[hg_launcher] create module with params {}".format(params))
    try:
        from . import hourglass_tiny
        HourglassModel = hourglass_tiny.HourglassModel
    except ImportError as e:
        util.warn(str(e))
        raise e
    model = HourglassModel(nFeat=params['nfeats'],
                           nStack=params['nstacks'],
                           nModules=params['nmodules'],
                           nLow=params['nlow'],
                           outputDim=params['num_joints'],
                           batch_size=params['batch_size'],
                           attention=params['mcam'],
                           training=do_training,
                           drop_rate=params['dropout_rate'],
                           lear_rate=params['learning_rate'],
                           decay=params['learning_rate_decay'],
                           decay_step=params['decay_step'],
                           dataset=dataset,
                           dataset_name=ds_name,
                           ckpt_dir=params['checkpoint_dir'],
                           logdir_train=join(params['checkpoint_dir'], 'training.log'),
                           logdir_test=join(params['checkpoint_dir'], 'testing.log'),
                           tiny=params['tiny'],
                           w_loss=params['weighted_loss'] or params['multichannel'],
                           joints= params['joint_list'],
                           modif=False,
                           use_fp16=params['fp16'])
    """
    Cached output
    """
    prediction_output = None if 'prediction_output' not in params else params['prediction_output']
    if not do_training:
        def cache_hit(prediction_output):
            if prediction_output is None:
                print(f"[hg_launcher][cache miss] prediction_output was not supplied as a parameter, cache is disabled")
                return False
            from . import matio
            new_hash = model.hash_saved_model(load=params['checkpoint_dir'], load_at=params['epoch_to_load'])
            if not isfile(prediction_output):
                print(f"[hg_launcher][cache miss] {prediction_output} does exist")
                return False
            d = matio.load(prediction_output)
            if 'MODEL_BLAKE2B' not in d:
                print(f"[hg_launcher][cache miss] {prediction_output} does not have model signature")
                return False
            old_hash = bytes(d['MODEL_BLAKE2B'])
            if old_hash != new_hash:
                print(f"[hg_launcher][cache miss] {prediction_output}'s model signature ({old_hash}) does not match our current model ({new_hash}), leaving")
                return False
            return True
        if cache_hit(prediction_output):
            print(f"[hg_launcher][cache hit] {prediction_output} is a valid cached model prediction, leaving")
            return

    model.generate_model()
    if do_training:
        # TODO: passing load= to continue if checkpoint presents
        # Alternatively we can let TF handles this.
        load_dir = params['checkpoint_dir'] if load else None
        util.ack('[launch_with_params] load dir {}'.format(load_dir))
        model.training_init(nEpochs=params['nepochs'],
                            epochSize=params['epoch_size'],
                            saveStep=params['saver_step'],
                            dataset=None,
                            load=load_dir,
                            continue_from=params['load_epoch'])
    else:
        out_dir = params['checkpoint_dir'] if 'output_dir' not in params else params['output_dir']
        model.testing_init(nEpochs=1, epochSize=params['prediction_epoch_size'], saveStep=0,
                           dataset=None,
                           load=params['checkpoint_dir'],
                           load_at=params['epoch_to_load'],
                           out_dir=out_dir,
                           prediction_output=prediction_output,
                           debug_predction=params['debug_predction'])

def main():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('hgcfg', help='Configure file of hourglass network')
    parser.add_argument('--training', help='Do training instead of testing')
    args = parser.parse_args()
    print('--Parsing Config File {}'.format(args.hgcfg))
    params = process_config(args.hgcfg)
    launch_with_params(params, do_training=args.training)

if __name__ == '__main__':
    main()
