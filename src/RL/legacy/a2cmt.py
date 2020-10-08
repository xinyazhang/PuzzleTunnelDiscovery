import a2c

class A2CTrainerMT(A2CTrainer):
    def __init__(self,
            envirs,
            advcore,
            tmax,
            gamma,
            learning_rate,
            queuemax=64,
            global_step=None,
            entropy_beta=0.1
            ):
        self.Q = queue.Queue(queuemax)
        self.threads = []
        for envir in envirs:

    def stop(self):
        for t in threads:
            self.Q.put(-1) # -1 means terminate
        for t in threads:
            t.join()
