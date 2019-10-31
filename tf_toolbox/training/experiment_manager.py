class ExperimentManager:
    def __init__(self,model_manager,optimizer_manager,*,logdir):
        self.model_manager = model_manager
        self.optimizer_manager = optimizer_manager

        self.hp_dict = dict(model_manager.hparam, **optimizer_manager.hparam)
        self.run_name = None
        self.logdir = logdir

    def prepare_run(self, run_name, **create_opts):
        self.optimizer_manager.create_optimizer(**create_opts)
        self.model_manager(optimizer=self.optimizer_manager.optimizer,**create_opts)
        self.run_name = run_name

    def do_run(self,**run_opts):
        if self.run_name is None:
            raise RuntimeError("No run initialized")

        hparams_values = dict([(self.hp_dict[k],run_opts[k]) for k in self.hp_dict])
        self.model_manager.train_model(logdir=self.logdir+"/"+self.run_name,hparams=hparams_values,**run_opts)