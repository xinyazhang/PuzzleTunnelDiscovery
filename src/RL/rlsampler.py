# SPDX-FileCopyrightText: Copyright © 2020 The University of Texas at Austin
# SPDX-FileContributor: Xinya Zhang <xinyazhang@utexas.edu>
# SPDX-License-Identifier: GPL-2.0-or-later
from curiosity import sess_no_hook
import curiosity
import pyosr
import numpy as np
from rlreanimator import reanimate
import uw_random
from six.moves import input
import progressbar
import rlcaction

def _press_enter():
    print("#############################################")
    print("##########CONGRATS TERMINAL REACHED##########")
    print("########## PRESS ENTER TO CONTINUE ##########")
    input("#############################################")

'''
Base class to visualize/evaluate the RL training results
'''
class RLVisualizer(object):
    def __init__(self, args, g, global_step):
        self.args = args
        self.dpy = pyosr.create_display()
        self.ctx = pyosr.create_gl_context(self.dpy)
        self.envir = curiosity.RigidPuzzle(args, 0)
        self.envir.egreedy = 0.995
        self.uw = self.envir.r
        self.r = self.envir.r
        #self.advcore = CuriosityRL(learning_rate=1e-3, args=args)
        self.advcore = curiosity.create_advcore(learning_rate=1e-3, args=args, batch_normalization=None)
        self.advcore.softmax_policy # Create the tensor
        self.gview = 0 if args.obview < 0 else args.obview
        if args.permutemag >= 0:
            self.envir.enable_perturbation(args.manual_p)
        self.mandatory_ckpt = True

    def attach(self, sess):
        self.sess = sess

class PolicyPlayer(RLVisualizer):
    def __init__(self, args, g, global_step):
        super(PolicyPlayer, self).__init__(args, g, global_step)

    def play(self):
        reanimate(self)

    def __iter__(self):
        envir = self.envir
        sess = self.sess
        advcore = self.advcore
        reaching_terminal = False
        pprefix = "[0] "
        while True:
            rgb,_ = envir.vstate
            yield rgb[self.gview] # First view
            if reaching_terminal:
                _press_enter()
                envir.reset()
            [policy, value] = advcore.evaluate([envir.vstate], sess, [advcore.softmax_policy, advcore.value])
            policy = policy[0][0]
            value = np.asscalar(value)
            action_index = advcore.make_decision(envir, policy, pprefix)
            print("Current Value {}".format(value))
            print("PolicyPlayer unmasked pol {}".format(policy))
            # print("PolicyPlayer masked pol {}".format(policy * advcore.action_mask))
            print("PolicyPlayer Action Index {}".format(action_index))
            nstate,reward,reaching_terminal,ratio = envir.peek_act(action_index, pprefix=pprefix)
            envir.qstate = nstate

class CriticPlayer(RLVisualizer):
    def __init__(self, args, g, global_step):
        super(CriticPlayer, self).__init__(args, g, global_step)

    def play(self):
        reanimate(self)

    def __iter__(self):
        envir = self.envir
        sess = self.sess
        advcore = self.advcore
        reaching_terminal = False
        pprefix = "[0] "
        while True:
            rgb,_ = envir.vstate
            yield rgb[self.gview] # First view
            if reaching_terminal:
                _press_enter()
                envir.reset()
            qs_bak = envir.qstate
            vs = []
            for a in self.args.actionset:
                nstate,reward,reaching_terminal,ratio = envir.peek_act(a)
                envir.qstate = nstate
                vs.append(envir.vstate)
            envir.qstate = qs_bak
            values = advcore.evaluate(vs, sess, [advcore.value])
            values = np.reshape(values, (-1))
            print("Values {}".format(values))
            ai = np.argmax(values)
            # print("PolicyPlayer unmasked pol {}".format(policy))
            # print("PolicyPlayer masked pol {}".format(policy * advcore.action_mask))
            # print("PolicyPlayer Action Index {}".format(action_index))
            nstate,reward,reaching_terminal,ratio = envir.peek_act(self.args.actionset[ai], pprefix=pprefix)
            envir.qstate = nstate

class QPlayer(RLVisualizer):
    def __init__(self, args, g, global_step):
        super(QPlayer, self).__init__(args, g, global_step)
        if args.permutemag > 0:
            self.envir.enable_perturbation()
        if args.samplein: # User can feed samples through samplein
            self.gt = np.load(args.samplein)
            self.gt_iter = 0
        else:
            self.gt = None

    def render(self, envir, state):
        envir.qstate = state
        return envir.vstate

    def play(self):
        if self.args.sampleout:
            self._sample()
        else:
            self._play()

    def _sample_mini_batch(self, batch):
        if self.gt is None:
            return [uw_random.gen_unit_init_state(envir.r) for i in range(args.batch)]
        states = np.take(self.gt['V'],
                indices=range(self.gt_iter, self.gt_iter + batch),
                axis=0, mode='wrap')
        self.gt_iter += batch
        return [state for state in states] # Convert 2D np.array to list of 1D np.array

    def _sample(self):
        Q = [] # list of states
        V = [] # list of numpy array of batched values
        args = self.args
        sess = self.sess
        advcore = self.advcore
        envir = self.envir
        assert args.iter % args.batch == 0, "presumably --iter is dividable by --batch"
        for i in range(args.iter/args.batch):
            states = self._sample_mini_batch(args.batch)
            Q += states
            images = [self.render(envir, state) for state in states]
            batch_rgb = [image[0] for image in images]
            batch_dep = [image[1] for image in images]
            dic = {
                    advcore.rgb_1: batch_rgb,
                    advcore.dep_1: batch_dep,
                  }
            values = sess.run(advcore.value, feed_dict=dic)
            values = np.reshape(values, [-1]) # flatten
            V.append(values)
        Q = np.array(Q)
        V = np.concatenate(V)
        np.savez(args.sampleout, Q=Q, V=V)

    def _play(self):
        reanimate(self)

    def __iter__(self):
        args = self.args
        sess = self.sess
        advcore = self.advcore
        envir = self.envir
        envir.reset()
        current_value = -1
        TRAJ = []
        while True:
            TRAJ.append(envir.qstate)
            yield envir.vstate[0][args.obview] # Only RGB
            NS = []
            images = []
            # R = []
            T = []
            TAU = []
            state = envir.qstate
            print("> Current State {}".format(state))
            for action in args.actionset:
                envir.qstate = state # IMPORTANT: Restore the state to unpeeked condition
                nstate, reward, terminal, ratio = envir.peek_act(action)
                envir.qstate = nstate
                NS.append(nstate)
                T.append(terminal)
                TAU.append(ratio)
                image = envir.vstate
                images.append(image)
            batch_rgb = [image[0] for image in images]
            batch_dep = [image[1] for image in images]
            dic = {
                    advcore.rgb_1: batch_rgb,
                    advcore.dep_1: batch_dep,
                  }
            values = sess.run(advcore.value, feed_dict=dic)
            values = np.reshape(values, [-1]) # flatten
            best = np.argmax(values, axis=0)
            print("> Current Values {}".format(values))
            print("> Taking Action {} RATIO {}".format(best, TAU[best]))
            print("> NEXT State {} Value".format(NS[best], values[best]))
            envir.qstate = NS[best]
            should_reset = False
            if current_value > values[best] or TAU[best] == 0.0:
                input("FATAL: Hit Local Maximal! Press Enter to restart")
                should_reset = True
            else:
                current_value = values[best]
            if T[best]:
                input("DONE! Press Enter to restart ")
                should_reset = True
            if should_reset:
                fn = input("Enter the filename to save the trajectory ")
                if fn:
                    TRAJ.append(envir.qstate)
                    TRAJ = np.array(TRAJ)
                    np.savez(fn, TRAJ=TRAJ, SINGLE_PERM=envir.get_perturbation())
                envir.reset()
                current_value = -1
                TRAJ = []

class CuriositySampler(RLVisualizer):
    def __init__(self, args, g, global_step):
        super(CuriositySampler, self).__init__(args, g, global_step)
        assert args.visualize == 'curiosity', '--visualize must be curiosity'
        assert args.curiosity_type == 1, "--curiosity_type should be 1 if --visualize is enabled"
        assert args.sampleout != '', '--sampleout must be enabled for --visualize curiosity'

    def play(self):
        args = self.args
        sess = self.sess
        advcore = self.advcore
        envir = self.envir
        samples= []
        curiosities_by_action = [ [] for i in range(uw_random.DISCRETE_ACTION_NUMBER) ]
        for i in range(args.iter):
            state = uw_random.gen_unit_init_state(envir.r)
            envir.qstate = state
            samples.append(state)
            for action in range(uw_random.DISCRETE_ACTION_NUMBER):
                nstate, reward, terminal, ratio = envir.peek_act(action)
                areward = advcore.get_artificial_reward(envir, sess,
                        state, action, nstate, ratio)
                curiosities_by_action[action].append(areward)
        samples = np.array(samples)
        curiosity = np.array(curiosities_by_action)
        np.savez(args.sampleout, Q=samples, C=curiosity)

class Fake3dSampler(RLVisualizer):
    def __init__(self, args, g, global_step):
        super(Fake3dSampler, self).__init__(args, g, global_step)
        assert args.sampleout, '--visualize fake3d requires --sampleout'
        self.istate = np.array(self.r.translate_to_unit_state(args.istateraw), dtype=np.float32)
        if 10 in args.actionset or 11 in args.actionset:
            self.has_rotation = True
        else:
            self.has_rotation = False
        self.zaxis_row = np.array([[0,0,1]],dtype=np.float32)

    def _sample(self, scale = 5.0):
        while True:
            quat = np.array([1, 0, 0, 0], dtype=np.float32)
            if self.has_rotation:
                theta = np.random.rand() * np.pi
                quat[0] = np.cos(theta) # * 0.5 was cancelled since we sampled from [0,pi)
                quat[3] = np.sin(theta) # rotation around z axis, hence [1,2] == 0
            tr = scale * (np.random.rand(3) - 0.5)
            tr[2] = self.istate[2] # Fix at the same plane
            state = np.concatenate((tr, quat))
            if self.r.is_disentangled(state):
                continue
            if self.r.is_valid_state(state):
                return state

    '''
    Sample in 2D space and convert it to 3D
    '''
    def _sample_mini_batch(self, batch):
        ret = []
        for i in range(batch):
            ret.append(self._sample())
        return ret

    def render(self, envir, state):
        envir.qstate = state
        return envir.vstate

    def play(self):
        args = self.args
        advcore = self.advcore
        envir = self.envir
        sess = self.sess
        Q = []
        V = []
        Po = []
        for i in progressbar.progressbar(range(args.iter / args.batch)):
            qbatch = self._sample_mini_batch(args.batch)
            images = [self.render(envir, state) for state in qbatch]
            batch_rgb = [image[0] for image in images]
            batch_dep = [image[1] for image in images]
            dic = {
                    advcore.rgb_1: batch_rgb,
                    advcore.dep_1: batch_dep,
                  }
            policies, values = sess.run([advcore.softmax_policy, advcore.value], feed_dict=dic)
            values = np.reshape(values, [-1]) # flatten
            policies = np.reshape(policies, [-1, advcore.action_space_dimension]) #
            Q += qbatch
            V.append(values)
            Po.append(policies)
        Q = np.array(Q)
        V = np.concatenate(V)
        Po = np.concatenate(Po)
        np.savez(args.sampleout, Q=Q, V=V, Po=Po)

'''
Milestome To State Action sampler
'''
class MSASampler(RLVisualizer):
    def __init__(self, args, g, global_step):
        super(MSASampler, self).__init__(args, g, global_step)
        assert args.msiraw, '--visualize msa requires --msiraw as input'
        assert args.sampleout, '--visualize msa requires --sampleout'
        msraw = np.array(args.msiraw)
        self.msraw = np.reshape(msraw, (-1, 7))
        assert self.msraw[0].all() == np.array(args.msiraw[:7]).all(), "--msiraw reshaped well"
        ms = []
        for msraw in self.msraw:
            ms.append(np.array(self.r.translate_to_unit_state(msraw), dtype=np.float32))
        self.ms = np.array(ms, dtype=np.float32)
        # self.traj_s = []
        # self.traj_a = []
        # self.traj_fv = []
        self.mandatory_ckpt = False

    def play(self):
        reanimate(self)

    def _fv(self):
        advcore = self.advcore
        envir = self.envir
        sess = self.sess
        return advcore.evaluate([envir.vstate], sess, advcore.model.cur_featvec)

    def __iter__(self):
        envir = self.envir
        sess = self.sess
        reaching_terminal = False
        args = self.args
        pprefix = "[0] "
        traj_s = [envir.qstate]
        traj_a = []
        traj_fv = [self._fv()]
        print("Initial state {}".format(envir.qstate))
        print("MS list {}".format(self.ms))
        rgb,_ = envir.vstate
        yield rgb[self.gview] # First view
        for ms in self.ms:
            print("Switch MS to {}".format(ms))
            while True:
                print("Current state {}".format(envir.qstate))
                sa = []
                sd = []
                rt = []
                rw = []
                for a in args.actionset:
                    nstate,r,reaching_terminal,_ = envir.peek_act(a)
                    sa.append(nstate)
                    sd.append(pyosr.distance(nstate, ms))
                    rw.append(r)
                    rt.append(reaching_terminal)
                print("sa {} \nsd {}".format(sa, sd))
                ai = np.argmin(sd)
                if sd[ai] > pyosr.distance(envir.qstate, ms): # Local minimum reached
                    break
                a = args.actionset[ai]
                traj_a.append(a)
                envir.qstate = sa[ai]
                traj_s.append(envir.qstate)
                traj_fv.append(self._fv())
                rgb,_ = envir.vstate
                if rw[ai] < 0:
                    print("#############################################")
                    print("!!!!!!!!!!   CRITICAL: COLLISION   !!!!!!!!!!")
                    print("!!!!!!!!!!   PRESS ENTER TO EXIT   !!!!!!!!!!")
                    input("#############################################")
                    np.savez(args.sampleout+'.die',
                             TRAJ_S=traj_s,
                             TRAJ_A=traj_a,
                             TRAJ_FV=traj_fv)
                    return
                    yield
                if rt[ai]:
                    _press_enter()
                    np.savez(args.sampleout,
                             TRAJ_S=traj_s,
                             TRAJ_A=traj_a,
                             TRAJ_FV=traj_fv)
                    return
                    yield
                yield rgb[self.gview] # First view

class CActionPlayer(RLVisualizer):
    def __init__(self, args, g, global_step, sancheck=False):
        assert args.samplein, '--samplein is mandatory for --visualize caction'
        super(CActionPlayer, self).__init__(args, g, global_step)
        d = np.load(args.samplein)
        self.V = d['V']
        self.N = d['N']
        self.D = d['D']
        self.mandatory_ckpt = False
        self.sancheck = sancheck

    def play(self):
        if self.sancheck:
            self.sanity_check()
        else:
            reanimate(self, fps=30)

    def __iter__(self):
        envir = self.envir
        uw = envir.r
        amag = self.args.amag
        V = self.V
        N = self.N
        D = self.D
        while True:
            if self.args.samplein2:
                VS = np.load(self.args.samplein2)['VS']
                vs = np.array([uw.translate_to_unit_state(v) for v in VS])
                for qs,crt,caa in rlcaction.trajectory_to_caction(vs, uw, amag):
                    envir.qstate = qs
                    rgb,_ = envir.vstate
                    yield rgb[self.gview]
            else:
                for qs,crt,caa in rlcaction.caction_generator(V, N, D, amag, uw):
                    assert envir.r.is_valid_state(qs)
                    envir.qstate = qs
                    rgb,_ = envir.vstate
                    yield rgb[self.gview] # First view
            assert envir.r.is_disentangled(envir.qstate)
            _press_enter()

    def sanity_check(self):
        envir = self.envir
        uw = envir.r
        amag = self.args.amag
        V = self.V
        N = self.N
        D = self.D
        for ivi in progressbar.progressbar(range(len(V))):
            # print("IVI {}".format(ivi))
            vs = rlcaction.ivi_to_leaving_trajectory(ivi, V, N, D, amag, uw)
            if not vs:
                continue
            for qs,crt,caa in rlcaction.trajectory_to_caction(vs, uw, amag):
                assert envir.r.is_valid_state(qs)

class CActionPlayer2(RLVisualizer):
    def __init__(self, args, g, global_step, sancheck=False):
        assert args.samplein, '--samplein is mandatory for --visualize caction2'
        super(CActionPlayer2, self).__init__(args, g, global_step)
        self.mandatory_ckpt = False
        VS = np.load(self.args.samplein)['VS']
        # print("ALLVS {}".format(VS))
        uw = self.envir.r
        self.known_path = [uw.translate_to_unit_state(v) for v in VS]
        self.out_index = 0

    def play(self):
        reanimate(self, fps=30)

    def __iter__(self):
        envir = self.envir
        uw = envir.r
        amag = self.args.amag
        out_dir = self.args.sampleout
        while True:
            QS = []
            CTR = []
            CAA = []
            for qs,ctr,caa in rlcaction.caction_generator2(uw, self.known_path, 0.15, amag):
            #for qs,crt,caa in rlcaction.trajectory_to_caction(self.known_path, uw, amag):
                QS.append(qs)
                CTR.append(ctr)
                CAA.append(caa)
                assert envir.r.is_valid_state(qs)
                # print('qs {}'.format(qs))
                # print('ctr {}'.format(ctr))
                # print('caa {}'.format(caa))
                envir.qstate = qs
                rgb,_ = envir.vstate
                yield rgb[self.gview]
            # assert envir.r.is_disentangled(envir.qstate)
            if out_dir:
                fn = '{}/{}.npz'.format(out_dir, self.out_index)
                '''
                print("QS {}".format(QS))
                print("CTR {}".format(CTR))
                print("CAA {}".format(CAA))
                print("lengths {} {} {}".format(len(QS), len(CTR), len(CAA)))
                print("### SAVING TO {}".format(fn))
                QS = np.array(QS)
                print("QS {}".format(QS))
                CTR = np.array(CTR)
                print("CTR {}".format(CTR))
                CAA = np.array(CAA)
                print("CAA {}".format(CAA))
                print('sizes {} {} {}'.format(QS.shape, CTR.shape, CAA.shape))
                '''
                np.savez(fn, QS=QS, CTR=CTR, CAA=CAA)
                print("### {} SAVED".format(fn))
                self.out_index += 1
                '''
                if self.args.iter > 0 and self.out_index > self.args.iter:
                    return
                '''
            else:
                _press_enter()

class CActionPlayer2ISFinder(RLVisualizer):
    def __init__(self, args, g, global_step, sancheck=False):
        assert args.samplein, '--samplein is mandatory for --visualize caction2_istate_finder'
        assert args.sampleout, '--sampleout is mandatory for --visualize caction2_istate_finder'
        super(CActionPlayer2ISFinder, self).__init__(args, g, global_step)
        self.mandatory_ckpt = False
        VS = np.load(self.args.samplein)['VS']
        # print("ALLVS {}".format(VS))
        uw = self.envir.r
        self.known_path = [uw.translate_to_unit_state(v) for v in VS]
        self.out_index = 0

class LocoPlayer(RLVisualizer):
    def __init__(self, args, g, global_step, sancheck=False):
        super(LocoPlayer, self).__init__(args, g, global_step)
        self.mandatory_ckpt = True
        self.verify_magnitude = args.vmag
        self.loco_out = self.advcore.polout[:,:,:6]

    def play(self):
        reanimate(self)

    def __iter__(self):
        envir = self.envir
        r = envir.r
        sess = self.sess
        advcore = self.advcore
        reaching_terminal = False
        while True:
            rgb,_ = envir.vstate
            yield rgb[self.gview] # First view
            if reaching_terminal:
                _press_enter()
                envir.reset()
            [loco] = advcore.evaluate([envir.vstate], sess, [self.loco_out])
            disp_pred = loco[0][0] # first batch first view for [B,V,...]
            cvm = self.verify_magnitude
            while cvm >= 1e-8:
                nstate, done, ratio = r.transit_state_by(envir.qstate,
                        disp_pred[:3], disp_pred[3:], cvm)
                print("[WIP] Next {} Done {} Ratio {} CVM {}".format(nstate, done, ratio, cvm))
                if ratio == 0.0:
                    cvm /= 2.0
                else:
                    break
            if ratio < 1e-8:
                ratio == 0.0
                nstate = np.copy(envir.qstate)
            print("Prediction {}".format(disp_pred))
            reaching_terminal = r.is_disentangled(nstate)
            envir.qstate = nstate

class TunnelFinder(RLVisualizer):
    def __init__(self, args, g, global_step):
        super(TunnelFinder, self).__init__(args, g, global_step)
        self.mandatory_ckpt = True
        '''
        Debugging Routines
        '''
        #TV = np.load('../res/alpha/alpha-1.2.org.tunnel.npz')['TUNNEL_V']
        #TV = np.load('alpha-1.2.org.tunnel.npz')['TUNNEL_V']
        TV = np.load('alpha-1.2.org.tunnel-more.npz')['TUNNEL_V']
        self.unit_tunnel_v = np.array([self.envir.r.translate_to_unit_state(v) for v in TV])
        self.sample_index = 0
        np.set_printoptions(precision=18)

        dt = args.amag
        da = dt

        self.delta_tr = np.array(
                [
                    [ dt, 0.0, 0.0],
                    [-dt, 0.0, 0.0],
                    [0.0,  dt, 0.0],
                    [0.0, -dt, 0.0],
                    [0.0, 0.0,  dt],
                    [0.0, 0.0, -dt],
                    [0.0, 0.0, 0.0],
                    [0.0, 0.0, 0.0],
                    [0.0, 0.0, 0.0],
                    [0.0, 0.0, 0.0],
                    [0.0, 0.0, 0.0],
                    [0.0, 0.0, 0.0],
                ])
        self.delta_aa = np.array(
                [
                    [0.0, 0.0, 0.0],
                    [0.0, 0.0, 0.0],
                    [0.0, 0.0, 0.0],
                    [0.0, 0.0, 0.0],
                    [0.0, 0.0, 0.0],
                    [0.0, 0.0, 0.0],
                    [ da, 0.0, 0.0],
                    [-da, 0.0, 0.0],
                    [0.0,  da, 0.0],
                    [0.0, -da, 0.0],
                    [0.0, 0.0,  da],
                    [0.0, 0.0, -da],
                ])
        self.cr_noneed = 0
        self.cr_success = 0
        self.cr_fail = 0
        self.cr_states = []
        self.to_cr_states = []
        self.cr_state_indicators = []

    def play(self):
        reanimate(self)

    def sample_generator(self):
        if self.args.iter > 0:
            for i in range(self.args.iter):
                yield i
        else:
            while True:
                yield

    def __iter__(self):
        envir = self.envir
        sess = self.sess
        advcore = self.advcore
        for _ in self.sample_generator():
            q = uw_random.random_state(0.75)
            envir.qstate = q
            if False:
                for i in range(len(self.unit_tunnel_v)):
                    envir.qstate = self.unit_tunnel_v[i]
                    yield envir.vstate[0][self.gview]
            DEBUG2 = False
            if DEBUG2:
                assert self.args.samplein, "DEBUG2=True requires --samplein"
                fn = '{}/{}.npz'.format(self.args.samplein, self.sample_index)
                print(fn)
                self.sample_index += 1
                d = np.load(fn)
                qs = d['QS']
                dqs = d['DQS']
                closes = d['CLOSES']
                for q,dq,close in zip(qs, dqs, closes):
                    #tr,aa,_ = pyosr.differential(q, close)
                    #dq = np.concatenate([tr, aa], axis=-1)
                    app = pyosr.apply(q, dq[:3], dq[3:])
                    print("q {}\ndq {}\nclose {}\napplied {}".format(q, dq, close, app))
                    print("diff {}\n".format(dq))
                    assert pyosr.distance(close, pyosr.apply(q, dq[:3], dq[3:])) < 1e-3
                    # continue
                    envir.qstate = q
                    yield envir.vstate[0][self.gview]
                    input("########## PRESS ENTER FOR APPLIED DQ ##########")
                    envir.qstate = app
                    yield envir.vstate[0][self.gview]
                    input("########## PRESS ENTER FOR CLOSEST Q ##########")
                    envir.qstate = close
                    yield envir.vstate[0][self.gview]
                    input("######## PRESS ENTER FOR THE NEXT TUPLE ########")
                continue # Skip the remaining
            DEBUG3 = False
            if DEBUG3:
                for q in self.unit_tunnel_v:
                    envir.qstate = q
                    yield envir.vstate[0][self.gview]
                continue
            DEBUG = False
            if DEBUG:
                yield envir.vstate[0][self.gview]
                distances = pyosr.multi_distance(q, self.unit_tunnel_v)
                ni = np.argmin(distances)
                close = self.unit_tunnel_v[ni]
                envir.qstate = close
                yield envir.vstate[0][self.gview]
                continue
            assert self.args.sampleout, '--sampleout is required for checking CR results'
            for i in range(3):
                vstate = envir.vstate
                yield vstate[0][self.gview]
                if envir.r.is_valid_state(envir.qstate):
                    print("===== Current State is Valid ====")
                    if i == 1: # Only capture the state after first movement
                        self.cr_noneed += 1
                else:
                    print("!!!!! Current State is not Valid ====")
                    print("!DEBUG UNIT: {}".format(envir.qstate))
                    nustate = envir.r.translate_from_unit_state(envir.qstate)
                    print("!DEBUG ORIGINAL: {}".format(nustate))
                    SEARCH_NUMERICAL = False
                    SEARCH_SA_FORCE = True
                    if SEARCH_NUMERICAL and i == 1: # Only capture the state after first movement
                        early_exit = False
                        current = envir.qstate
                        self.to_cr_states.append(current)
                        for k in range(32):
                            nexts = []
                            for tr,aa in zip(self.delta_tr, self.delta_aa):
                                nexts.append(pyosr.apply(current, tr, aa))
                            nexts = np.array(nexts)
                            SAs = envir.r.intersection_region_surface_areas(nexts, True)
                            print(SAs)
                            min_index = np.argmin(SAs)
                            current = nexts[min_index]
                            envir.qstate = current
                            vstate = envir.vstate
                            yield vstate[0][self.gview]
                            if envir.r.is_valid_state(envir.qstate):
                                early_exit = True
                                break
                        if early_exit:
                            self.cr_success += 1
                            self.cr_state_indicators.append(1)
                        else:
                            self.cr_fail += 1
                            self.cr_state_indicators.append(-1)
                        self.cr_states.append(envir.qstate)
                        self.sample_index += 1
                    if SEARCH_SA_FORCE and i == 1: # Force from optimizing surface area
                        # TODO: Unfinished, just print the force mag and direction
                        tup = envir.r.force_from_intersecting_surface_area(envir.qstate)
                        print('Force Pos:\n{}'.format(tup[0]))
                        print('Force Direction:\n{}'.format(tup[1]))
                        print('Force Mag:\n{}'.format(tup[2]))
                dic = {
                        advcore.rgb_tensor: [vstate[0]],
                        advcore.dep_tensor: [vstate[1]],
                      }
                disp_pred = sess_no_hook(sess, advcore.finder_pred, feed_dict=dic)
                disp_pred = disp_pred[0][0] # first batch first view for [B,V,...]

                nstate = pyosr.apply(envir.qstate, disp_pred[:3], disp_pred[3:])
                distances = pyosr.multi_distance(q, self.unit_tunnel_v)
                ni = np.argmin(distances)
                close = self.unit_tunnel_v[ni]
                # nstate = envir.r.translate_to_unit_state(nstate_raw)
                print("Prediction {}".format(disp_pred))
                print("Next (Unit) {}".format(nstate))
                print("Error {}".format(pyosr.distance(nstate, close)))
                envir.qstate = nstate
                # input("########## PRESS ENTER TO CONTINUE ##########")
        print("CR No Need {}\nCR SUCCESS {}\nCR FAIL{}".format(self.cr_noneed, self.cr_success, self.cr_fail))
        np.savez(self.args.sampleout, CRQS=self.cr_states, TOCRQS=self.to_cr_states, CRI=self.cr_state_indicators)

def create_visualizer(args, g, global_step):
    if args.qlearning_with_gt:
        # assert args.sampleout, "--sampleout is required to store the samples for --qlearning_with_gt"
        assert args.iter > 0, "--iter needs to be specified as the samples to generate"
        # assert False, "Evaluating of Q Learning is not implemented yet"
        return QPlayer(args, g, global_step)
    elif args.visualize == 'policy':
        return PolicyPlayer(args, g, global_step)
    elif args.visualize == 'critic':
        return CriticPlayer(args, g, global_step)
    elif args.visualize == 'curiosity':
        return CuriositySampler(args, g, global_step)
    elif args.visualize == 'fake3d':
        return Fake3dSampler(args, g, global_step)
    elif args.visualize == 'msa':
        return MSASampler(args, g, global_step)
    elif args.visualize == 'caction':
        return CActionPlayer(args, g, global_step)
    elif args.visualize == 'caction_sancheck':
        return CActionPlayer(args, g, global_step, sancheck=True)
    elif args.visualize == 'caction2':
        return CActionPlayer2(args, g, global_step)
    elif args.visualize == 'caction2_istate_finder':
        assert False, 'caction2_istate_finder not implemented yet'
        return CActionPlayer2ISFinder(args, g, global_step)
    elif args.visualize == 'loco':
        return LocoPlayer(args, g, global_step)
    elif args.visualize == 'tunnel_finder':
        return TunnelFinder(args, g, global_step)
    assert False, '--visualize {} is not implemented yet'.format(args.visualize)
