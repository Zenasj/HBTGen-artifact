import tensorflow as tf

class SaveBestCheckpointSaverListener(tf.train.CheckpointSaverListener):
    def __init__(self, save_checkpoints_steps, keep_checkpoint_max, output_dir):
        self.best_ckpt = None
        self.best_path_match = None
        self.best_global_step_value = None
        self.save_checkpoints_steps = save_checkpoints_steps
        self.keep_checkpoint_max = keep_checkpoint_max
        self.output_dir = output_dir

    def begin(self):
        # You can add ops to the graph here.
        print('Starting the session.')
        # self.your_tensor = ...

    def before_save(self, session, global_step_value):
        print('About to write a checkpoint')

    def after_save(self, session, global_step_value):
        print('Done writing checkpoint at {}.'.format(global_step_value))
        global_step_value = int(global_step_value)
        if global_step_value == 0:
            return
        current_ckpt = 'model.ckpt-{}'.format(global_step_value)
       # do beam search prediction
        current_path_match = do_predict()
        print('current result: {} : {}'.format(current_ckpt, current_path_match))
        if not self.best_ckpt:
            self.best_ckpt = current_ckpt
            self.best_path_match = current_path_match
            self.best_global_step_value = str(global_step_value)
        else:
            if current_path_match > self.best_path_match:
                self.best_ckpt = current_ckpt
                self.best_path_match = current_path_match
                self.best_global_step_value = str(global_step_value)
                print('Saved best ckpt with path_match {}'.format(current_path_match))

    def end(self, session, global_step_value):
        print('best model {}, remove useless models.'.format(self.best_ckpt))
        # for f in os.listdir(self.output_dir):
        #     file = os.path.join(self.output_dir, f)
        #     if os.path.isfile(file) and f.startswith('model'):
        #         model_global_step = f[f.index('-') + 1:f.rindex('.')]
        #         if model_global_step != self.best_global_step_value:
        #             os.remove(file)
        #             print('remove {}'.format(file))