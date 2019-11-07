#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from os.path import join, abspath, expanduser
import subprocess
import pathlib
import numpy as np
import h5py
import os
try:
    from progressbar import progressbar
except ImportError:
    progressbar = lambda x: x

from . import util
from . import choice_formatter
from . import matio
from . import partt
from . import touchq_util
from . import parse_ompl
from . import condor

hdf5_overwrite = matio.hdf5_overwrite

# Pipeline local file
#
# Here we always use "trajectory" to represent the curve in configuration space
# and "path" always refers to the path in a file system.
_TRAJECTORY_SCRATCH = util.PREP_TRAJECTORY_SCRATCH
_KEYCAN_SCRATCH = util.PREP_KEY_CAN_SCRATCH

def _get_candidate_file(ws):
    return ws.local_ws(util.MT_KEY_CANDIDATE_FILE) + '.xz'

'''
System Architecture:
    1. All tasks are executed locally, which makes debugging easier
    2. autorun() will ssh into remote hosts to do the corresponding tasks (e.g. submit condor jobs)
    3. Synchronous Condor job is achieved through condor_submit and condor_wait
    4. If the ssh connection is broken while condor_wait,
       autorun will try it again after 5 seconds
'''

def find_trajectory(args, ws):
    util.log('[find_trajectory] args {}'.format(args))
    se3solver_path = ws.condor_local_exec('se3solver.py')
    scratch_rel = _TRAJECTORY_SCRATCH
    scratch_dir = ws.local_ws(scratch_rel)
    _, config = parse_ompl.parse_simple(ws.training_puzzle)
    condor_args = [se3solver_path,
                   'solve', ws.training_puzzle,
                   '--cdres', config.getfloat('problem', 'collision_resolution', fallback=0.0001),
                   '--trajectory_out', '{}/traj_$(Process).npz'.format(scratch_dir),
                   ws.config.get('TrainingTrajectory', 'PlannerAlgorithmID'),
                   ws.config.get('TrainingTrajectory', 'CondorTimeThreshold'),
                   ]
    if args.only_wait:
        condor.local_wait(scratch_dir)
        return
    condor.local_submit(ws,
                        util.PYTHON,
                        iodir_rel=scratch_rel,
                        arguments=condor_args,
                        instances=ws.config.get('TrainingTrajectory', 'CondorInstances'),
                        wait=True)


def interpolate_trajectory(args, ws):
    import pyosr
    scratch_dir = ws.local_ws(_TRAJECTORY_SCRATCH)
    candidate_file = ws.local_ws(util.MT_KEY_CANDIDATE_FILE)
    # Do not process only_wait, we need to restart if ssh is broken
    Qs = []
    DEBUG = False
    if DEBUG:
        uw = util.create_unit_world(ws.local_ws(util.TRAINING_DIR, util.PUZZLE_CFG_FILE))
        util.log("[DEBUG] Reference 21.884796142578125 17.07219123840332 2.7253246307373047")
    traj_files = sorted(pathlib.Path(scratch_dir).glob("traj_*.npz"))
    ntraj = ws.config.getint('TrainingKeyConf', 'TrajectoryLimit', fallback=-1)
    if ntraj < 0:
        ntraj = len(traj_files)
    f = matio.hdf5_safefile(candidate_file)
    for i, fn in enumerate(traj_files[:ntraj]):
        d = matio.load(fn)
        if d['FLAG_IS_COMPLETE'] == 0:
            continue
        util.log('[interpolate_trajectory] interpolating trajectory {}'.format(fn))
        traj = d['OMPL_TRAJECTORY']
        metrics = pyosr.path_metrics(traj)
        npoint = ws.config.getint('TrainingKeyConf', 'CandidateNumber')
        dtau = metrics[-1] / float(npoint)
        Qs = [pyosr.path_interpolate(traj, metrics, dtau * i) for i in range(npoint)]
        if DEBUG:
            raise NotImplementedError() # TODO: Update the debugging code or replace it
            utraj = uw.translate_ompl_to_unit(traj)
            dout = fn.stem+'.unit.txt'
            matio.savetxt(dout, utraj)
            util.log('[DEBUG][interpolate_trajectory] dump unitary path to {}'.format(dout))
        traj_name = pathlib.Path(fn).stem
        traj_id = int(traj_name[len('traj_'):])
        hdf5_overwrite(f, 'traj_{}'.format(util.padded(traj_id, ntraj)), Qs)
    util.log('[interpolate_trajectory] saving the interpolation results to {}'.format(candidate_file))
    #np.savez(candidate_file, OMPL_CANDIDATES=Qs)
    f.close()
    util.xz(candidate_file)


'''
Note: clearance is measured in unitary configuration space
'''
def estimate_clearance_volume(args, ws):
    scratch_dir = ws.local_ws(_KEYCAN_SCRATCH)
    os.makedirs(scratch_dir, exist_ok=True)
    if args.only_wait:
        condor.local_wait(scratch_dir)
        return
    # Prepare the data and task description
    candidate_file = _get_candidate_file(ws)
    util.log('[estimate_clearance_volume] loading file {}'.format(candidate_file))
    cf = matio.load(candidate_file)
    trajs = sorted(list(cf.keys()))
    ntraj = len(trajs)
    nq = cf[trajs[0]].shape[0]
    #uw = ws.condor_unit_world(util.TRAINING_DIR)
    uw = util.create_unit_world(ws.local_ws(util.TRAINING_DIR, util.PUZZLE_CFG_FILE))
    DEBUG = False
    if DEBUG:
        unit_qs = uw.translate_ompl_to_unit(Qs)
        for uq in progressbar(unit_qs):
            assert uw.is_valid_state(uq)
        return
    task_shape = (ntraj, nq)
    total_chunks = partt.guess_chunk_number(task_shape,
            ws.config.getint('SYSTEM', 'CondorQuota') * 2,
            ws.config.getint('TrainingKeyConf', 'ClearanceTaskGranularity'))
    if args.task_id is None:
        util.log('[estimate_clearance_volume] task shape {}'.format(task_shape))
        util.log('[estimate_clearance_volume] partitioned into {} chunks'.format(total_chunks))

    if total_chunks > 1 and args.task_id is None:
        os.makedirs(scratch_dir, exist_ok=True)
        # Submit a Condor job
        condor_args = [ws.condor_local_exec('facade.py'),
                       'preprocess_key',
                       '--stage', 'estimate_clearance_volume',
                       '--task_id', '$(Process)',
                       ws.local_ws()]
        condor.local_submit(ws,
                            util.PYTHON,
                            iodir_rel=_KEYCAN_SCRATCH,
                            arguments=condor_args,
                            instances=total_chunks,
                            wait=True)
    else:
        # Do the actual work (either no need to partition, or run from HTCondor)
        task_id = 0 if total_chunks == 1 else args.task_id
        tindices = partt.get_task_chunk(task_shape, total_chunks, task_id)
        npoint = ws.config.getint('TrainingKeyConf', 'ClearanceSample')
        # NOTE: THE COMMA, TODO: check all loops that's using get_task_chunk
        cached_traj = None
        batch_str = util.padded(task_id, total_chunks)
        out_fn = ws.local_ws(scratch_dir, 'unitary_clearance_from_keycan-batch_{}.hdf5'.format(batch_str))
        f = matio.hdf5_safefile(out_fn)
        # tindices = tindices[:4]
        for traj_id, qi in progressbar(tindices):
            traj_name = trajs[traj_id]
            if cached_traj != traj_id:
                Qs = cf[traj_name]
                nq = Qs.shape[0]
                unit_qs = uw.translate_ompl_to_unit(Qs)
                cached_traj = traj_id
            free_vertices, touch_vertices, to_inf, free_tau, touch_tau = touchq_util.calc_touch(uw, unit_qs[qi], npoint, uw.recommended_cres)
            # out_fn = join(scratch_dir, 'unitary_clearance_from_keycan-{}.npz'.format(qi_str))
            qi_str = util.padded(qi, nq)
            gpn = traj_name + '/' + qi_str + '/'
            hdf5_overwrite(f, gpn+'FROM_V_OMPL', Qs[qi])
            hdf5_overwrite(f, gpn+'FROM_V', unit_qs[qi])
            hdf5_overwrite(f, gpn+'FREE_V', free_vertices)
            hdf5_overwrite(f, gpn+'TOUCH_V', touch_vertices)
            hdf5_overwrite(f, gpn+'IS_INF', to_inf)
            hdf5_overwrite(f, gpn+'FREE_TAU', free_tau)
            hdf5_overwrite(f, gpn+'TOUCH_TAU', touch_tau)
            '''
            np.savez_compressed(out_fn,
                     FROM_V_OMPL=Qs[qi],
                     FROM_V=unit_qs[qi],
                     FREE_V=free_vertices,
                     TOUCH_V=touch_vertices,
                     IS_INF=to_inf,
                     FREE_TAU=free_tau,
                     TOUCH_TAU=touch_tau)
            '''
        f.close()
        util.xz(out_fn)

def pickup_key_configuration_old(args, ws):
    import pyosr
    scratch_dir = ws.local_ws(_KEYCAN_SCRATCH)
    os.makedirs(scratch_dir, exist_ok=True)
    top_k = ws.config.getint('TrainingKeyConf', 'KeyConf')
    fn_list = []
    # Do not process only_wait, we need to restart if ssh is broken
    for fn in pathlib.Path(scratch_dir).glob('unitary_clearance_from_keycan-*.npz'):
        fn_list.append(fn)
    fn_list = sorted(fn_list)
    median_list = []
    max_list = []
    min_list = []
    mean_list = []
    stddev_list = []
    for fn in progressbar(fn_list[:]):
        d = matio.load(fn)
        distances = pyosr.multi_distance(d['FROM_V'], d['FREE_V'])
        median_list.append(np.median(distances))
        max_list.append(np.max(distances))
        min_list.append(np.min(distances))
        mean_list.append(np.mean(distances))
        stddev_list.append(np.std(distances))
    # mean works better than median, for some reason
    top_k_indices = np.array(mean_list).argsort()[:top_k]
    util.log('[pickup_key_configuration] top k {}'.format(top_k_indices))
    kq_ompl = []
    kq = []
    for k in top_k_indices:
        fn = fn_list[k]
        d = matio.load(fn)
        kq_ompl.append(d['FROM_V_OMPL'])
        kq.append(d['FROM_V'])
    key_out = ws.local_ws(util.KEY_FILE)
    stat_out = np.array([median_list, max_list, min_list, mean_list, stddev_list])
    util.log('[pickup_key_configuration] writting results to {}'.format(key_out))
    np.savez(key_out, KEYQ_OMPL=kq_ompl, KEYQ=kq, _STAT=stat_out, _TOP_K=top_k_indices)
    util.xz(key_out)

def pickup_key_configuration(args, ws):
    import pyosr
    uw = util.create_unit_world(ws.local_ws(util.TRAINING_DIR, util.PUZZLE_CFG_FILE))

    scratch_dir = ws.local_ws(_KEYCAN_SCRATCH)
    os.makedirs(scratch_dir, exist_ok=True)
    '''
    Pick up top_k from each trajectory
    '''
    top_k = ws.config.getint('TrainingKeyConf', 'KeyConf')
    candidate_file = _get_candidate_file(ws)
    cf = matio.load(candidate_file)
    trajs = sorted(list(cf.keys()))
    ntraj = len(trajs)
    util.log('[pickup_key_configuration] # of trajs {}'.format(ntraj))
    util.log('[pickup_key_configuration] actual trajs {}'.format(trajs))
    nq = cf[trajs[0]].shape[0]
    fn_list = sorted(pathlib.Path(scratch_dir).glob('unitary_clearance_from_keycan-batch_*.hdf5.xz'))
    # fn_list = fn_list[:3] # Debug
    '''
    Load distances to traj_mean_list (dict of dict of index to list)
    '''
    # CAVEAT: stat_out may not be continuous
    # stat_out = np.full((ntraj, nq, 5), np.finfo(np.float64).max, dtype=np.float64)
    stat_dict = {} # np.full((ntraj, nq, 5), np.finfo(np.float64).max, dtype=np.float64)
    for fn in progressbar(fn_list):
        # util.log("loading {}".format(fn)) # Debug
        d = matio.load(fn)
        # trajectory level
        for traj_name in d.keys():
            traj_id = int(traj_name[len('traj_'):])
            traj_grp = d[traj_name]
            for q_name in traj_grp.keys():
                q = traj_grp[q_name]
                if False:
                    # FIMXE: this distance is buggy
                    distances = uw.multi_kinetic_energy_distance(q['FROM_V'], q['FREE_V'])
                else:
                    distances = pyosr.multi_distance(q['FROM_V'], q['FREE_V'])
                idx = int(str(q_name))
                # util.log("checking traj_name {} traj_id {} idx {}".format(traj_name, traj_id, idx)) # Debug
                m = np.mean(distances)
                if traj_id not in stat_dict:
                    stat_dict[traj_id] = np.full((nq, 5), np.finfo(np.float64).max, dtype=np.float64)
                stat_dict[traj_id][idx] = np.array([np.median(distances), np.max(distances), np.min(distances), m, np.std(distances)])
                # util.log("checking traj_name {} traj_id {} idx {} mean {}".format(traj_name, traj_id, idx, m)) # Debug
    kq_ompl = []
    top_k_indices = []
    for traj_name in progressbar(trajs):
        # util.log("checking traj_name {}".format(traj_name)) # Debug
        traj_id = int(traj_name[len('traj_'):])
        stats = stat_dict[traj_id]
        current_top_k_indices = np.array(stats[:,3]).argsort()[:top_k]
        # util.log("stat of traj_name {} traj_id {} top_k {}".format(traj_name, traj_id, current_top_k_indices)) # Debug
        for k in current_top_k_indices:
            kq_ompl.append(cf[traj_name][k])
            top_k_indices.append([traj_id, k])
    kq = uw.translate_ompl_to_unit(kq_ompl)
    key_out = ws.local_ws(util.KEY_FILE)
    # np.savez(key_out, KEYQ_OMPL=kq_ompl, KEYQ=kq, _STAT=stat_out, _TOP_K_PER_TRAJ=top_k_indices)
    stat_keys = []
    stat_values = []
    for key, value in stat_dict.items():
        stat_keys.append(key)
        stat_values.append(value)
    np.savez(key_out, KEYQ_OMPL=kq_ompl, KEYQ=kq, _STAT_KEYS=stat_keys, _STAT_VALUES=stat_values, _TOP_K_PER_TRAJ=top_k_indices)
    util.ack('[pickup_key_configuration] key configurations are written to {}'.format(key_out))

# We decided not to use reflections because we may leak internal functions to command line
function_dict = {
        'find_trajectory' : find_trajectory,
        'interpolate_trajectory' : interpolate_trajectory,
        'estimate_clearance_volume' : estimate_clearance_volume,
        'pickup_key_configuration' : pickup_key_configuration,
}

def setup_parser(subparsers):
    p = subparsers.add_parser('preprocess_key',
            help='Preprocessing step, to figure out the key configuration in the training puzzle',
            formatter_class=choice_formatter.Formatter)
    p.add_argument('--stage',
                   choices=list(function_dict.keys()),
                   help='R|Possible stages:\n'+'\n'.join(list(function_dict.keys())),
                   default='',
                   metavar='')
    p.add_argument('--only_wait', action='store_true')
    p.add_argument('--task_id', help='Feed $(Process) from HTCondor', type=int, default=None)
    p.add_argument('dir', help='Workspace directory')

def run(args):
    if args.stage in function_dict:
        ws = util.Workspace(args.dir)
        function_dict[args.stage](args, ws)
    else:
        print("Unknown preprocessing pipeline stage {}".format(args.stage))

# Hint: it's recommended to have a _remote_command per module
# Workspace.remote_command is too tedious to use directly.
def _remote_command(ws, cmd, auto_retry=True):
    ws.remote_command(ws.condor_host,
                      ws.condor_exec(),
                      ws.condor_ws(),
                      'preprocess_key', cmd, auto_retry=auto_retry)

def remote_find_trajectory(ws):
    _remote_command(ws, 'find_trajectory')

def remote_interpolate_trajectory(ws):
    _remote_command(ws, 'interpolate_trajectory')

def remote_estimate_clearance_volume(ws):
    _remote_command(ws, 'estimate_clearance_volume')

def remote_pickup_key_configuration(ws):
    _remote_command(ws, 'pickup_key_configuration')

def autorun(args):
    ws = util.Workspace(args.dir)
    ws.verify_training_puzzle()
    ws.deploy_to_condor(util.WORKSPACE_SIGNATURE_FILE,
                        util.WORKSPACE_CONFIG_FILE,
                        util.CONDOR_TEMPLATE,
                        util.TRAINING_DIR+'/')
    remote_find_trajectory(ws)
    remote_interpolate_trajectory(ws)
    remote_estimate_clearance_volume(ws)
    remote_pickup_key_configuration(ws)

def collect_stages():
    return [ ('deploy_to_condor',
              lambda ws: ws.deploy_to_condor(util.WORKSPACE_SIGNATURE_FILE,
                                             util.WORKSPACE_CONFIG_FILE,
                                             util.CONDOR_TEMPLATE,
                                             util.TRAINING_DIR+'/')
             ),
             ('find_trajectory', remote_find_trajectory),
             ('interpolate_trajectory', remote_interpolate_trajectory),
             ('estimate_clearance_volume', remote_estimate_clearance_volume),
             ('pickup_key_configuration', remote_pickup_key_configuration),
           ]
