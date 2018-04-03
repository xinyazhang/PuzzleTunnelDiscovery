import tensorflow as tf
import rlenv
import numpy as np

class A2CTrainer:
    a2c_tmax = None
    optimizer = None
    loss = None
    verbose_training = False

    def __init__(self,
            envir,
            advcore,
            tmax,
            gamma,
            learning_rate,
            global_step=None,
            entropy_beta=0.01
            ):
        self.envir = envir
        self.advcore = advcore
        self.a2c_tmax = tmax
        self.gamma = gamma
        self.entropy_beta = entropy_beta
        self.action_space_dimension = int(advcore.policy.shape[-1])
        '''
        Create the optimizers to train the AdvCore
        '''
        self.optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
        self.loss = self.build_loss(advcore)
        '''
        Do not train Vision since we don't have reliable GT from RL procedure
        '''
        self.train_op = self.optimizer.minimize(self.loss,
                global_step=global_step,
                var_list=advcore.policy_params + advcore.value_params + advcore.lstm_params)

    '''
    Private: Return A2C Loss

    Side effect: self.rl_params were set
    '''
    def build_loss(self, advcore):
        if self.loss is not None:
            return self.loss

        '''
        Input tensor of Ground Truth from Environment
        '''
        self.Adist_tensor = tf.placeholder(tf.float32,
                shape=[None, 1, self.action_space_dimension],
                name='ADistPh')
        self.TD_tensor = tf.placeholder(tf.float32, shape=[None], name='TDPh')
        self.V_tensor = tf.placeholder(tf.float32, shape=[None], name='VPh')

        policy = tf.nn.softmax(logits=advcore.policy)
        log_policy = tf.log(tf.clip_by_value(policy, 1e-20, 1.0))
        # cond_prob = tf.reduce_sum(policy * self.Adist_tensor, axis=1)
        action_entropy = tf.reduce_sum(tf.multiply(log_policy, self.Adist_tensor),
                reduction_indices=[1,2])
        entropy = -tf.reduce_sum(policy * log_policy, reduction_indices=[1,2])

        policy_loss_per_step = tf.reduce_sum(action_entropy * self.TD_tensor) + entropy * self.entropy_beta
        policy_loss = tf.reduce_sum(policy_loss_per_step)
        value_loss = tf.nn.l2_loss(self.V_tensor - advcore.value)
        self.loss = policy_loss+value_loss
        return self.loss

    '''
    Train the network

    This method interacts with RLEnv object to collect truths
    '''
    def train(self, sess, tmax=-1):
        if tmax < 0:
            tmax = self.a2c_tmax
        envir = self.envir
        advcore = self.advcore
        reaching_terminal = False
        states = []
        actions = []
        rewards = []
        values = []
        lstm_begin = advcore.get_lstm()
        for i in range(tmax):
            policy, value = advcore.evaluate_current(envir, sess, [advcore.policy, advcore.value])
            '''
            Pick up the only frame
            '''
            print('pol {} shape {}; val {} shape {}'.format(policy, policy.shape, value, value.shape))
            policy = policy[0][0]
            value = value[0][0][0]
            lstm_next = advcore.get_lstm()
            action = advcore.make_decision(policy)
            print('Action chosen {}'.format(action))
            states.append(envir.vstate)
            '''
            FIXME: Wait, shouldn't be policy?
            '''
            actions.append(action)
            values.append(value)

            print("Peeking action")
            nstate,reward,reaching_terminal = envir.peek_act(action)
            # print("action peeked {} ratio {} terminal? {}".format(nstate, ratio, reaching_terminal))
            adist = np.zeros(shape=(self.action_space_dimension),
                    dtype=np.float32)
            adist[action] = 1.0
            reward += advcore.get_artificial_reward(envir, sess, envir.qstate, adist, nstate)
            rewards.append(reward)
            if reaching_terminal:
                break
            advcore.set_lstm(lstm_next) # AdvCore next frame
            envir.qstate = nstate # Envir Next frame
        advcore.set_lstm(lstm_begin)
        self.a2c(sess, actions, states, rewards, values, reaching_terminal)
        advcore.set_lstm(lstm_next)

        if reaching_terminal:
            envir.reset()

    '''
    Private function that performs the training
    '''
    def a2c(self, sess, actions, states, rewards, values, reaching_terminal):
        envir = self.envir
        advcore = self.advcore
        V = 0.0
        if not reaching_terminal:
            V = np.asscalar(advcore.evaluate_current(envir, sess, tensors=[advcore.value])[0])

        states.reverse()
        actions.reverse()
        rewards.reverse()
        values.reverse()

        batch_rgb = []
        batch_dep = []
        batch_adist = []
        batch_td = []
        batch_V = []
        '''
        Calculate the per-step "true" value for current iteration
        '''
        '''
        print('[{}] R start with {}'.format(self.worker_thread_index, R))
        '''
        for (ai, ri, si, Vi) in zip(actions, rewards, states, values):
            V = ri + self.gamma * V
            td = V - Vi
            print("V {} Vi {}".format(V, Vi))
            adist = np.zeros(shape=(1, self.action_space_dimension),
                    dtype=np.float32)
            adist[0, ai] = 1.0
            [rgb,dep] = si

            batch_rgb.append(rgb)
            batch_dep.append(dep)
            batch_adist.append(adist)
            batch_td.append(td)
            batch_V.append(V)
        batch_rgb.append(envir.vstate[0])
        batch_dep.append(envir.vstate[1])
        if self.verbose_training:
            print('[{}] batch_a[0] {}'.format(self.worker_thread_index, batch_adist[0]))
            print('[{}] batch_V {}'.format(self.worker_thread_index, batch_R))
        '''
        Always reverse, the RLEnv need this sequential info for training.
        '''
        batch_rgb.reverse()
        batch_dep.reverse()
        batch_adist.reverse()
        batch_td.reverse()
        batch_V.reverse()
        dic = {
                advcore.rgb_1: batch_rgb[:-1],
                advcore.dep_1: batch_dep[:-1],
                self.Adist_tensor: batch_adist,
                self.TD_tensor: batch_td,
                self.V_tensor: batch_V
              }
        if advcore.using_lstm:
            dic.update({
                advcore.lstm_states_in.c : advcore.current_lstm.c,
                advcore.lstm_states_in.h : advcore.current_lstm.h,
                advcore.lstm_len : len(batch_rgb[:-1])
                       })
        print('batch_td {}'.format(batch_td))
        print('batch_V {}'.format(batch_V))
        sess.run(self.train_op, feed_dict=dic)
        advcore.train(sess, batch_rgb, batch_dep, batch_adist)
