import tensorflow as tf

#@title DNN: context.py

class CLVFeatures(object):

  HEADERS = ['customer_id', 'monetary_dnn', 'monetary_btyd', 'frequency_dnn',
             'frequency_btyd', 'recency', 'T', 'time_between',
             'avg_basket_value', 'avg_basket_size', 'cnt_returns',
             'has_returned', 'frequency_btyd_clipped', 'monetary_btyd_clipped',
             'target_monetary_clipped', 'target_monetary']
  HEADERS_DEFAULT = [[''], [0.0], [0.0], [0],
                     [0], [0], [0], [0.0],
                     [0.0], [0.0], [0],
                     [-1], [0], [0.0],
                     [0.0], [0.0]]
  NUMERICS = {
      'monetary_dnn': [],
      'recency': [],
      'frequency_dnn': [],
      'T': [],
      'time_between': [],
      'avg_basket_value': [],
      'avg_basket_size': [],
      'cnt_returns': []}
  CATEGORICALS_W_LIST = {
      'has_returned': [0, 1]}
  CROSSED = []
  KEY = 'customer_id'
  UNUSED = [KEY, 'monetary_btyd', 'frequency_btyd', 'frequency_btyd_clipped',
            'monetary_btyd_clipped', 'target_monetary_clipped']
  TARGET_NAME = 'target_monetary'

  def __init__(self, ignore_crosses=False): #, is_dnn=None

    #if not is_dnn:
    #  return

    self.ignore_crosses = ignore_crosses
    (self.headers, self.numerics_names, self.categorical_names) = self._keep_used()
    self.continuous, self.categorical = self._make_base_features()

    if not self.ignore_crosses:
      self.crossed_for_wide, self.crossed_for_deep = self._make_crossed()
  
  def _keep_used(self):
    headers = [h for h in self.HEADERS if h not in self.UNUSED]
    numerics_names = {
        k: v for k, v in iteritems(self.NUMERICS)
        if (k not in self.UNUSED) and (k != self.TARGET_NAME)
    }
    categorical_names = {
        k: v for k, v in iteritems(self.CATEGORICALS_W_LIST)
        if k not in self.UNUSED    
    }                          
    return headers, numerics_names, categorical_names
  
  def get_key(self):
    return self.KEY
  
  def get_used_headers(self, with_key=False, with_target=False):
    used_headers = [h for h in self.headers if h != self.TARGET_NAME]

    if with_key:
      used_headers.insert(0, self.KEY)
    if with_target:
      used_headers.append(self.TARGET_NAME)

    return used_headers

  def get_defaults(self, headers_names=None, with_key=False):
    if headers_names is None:
      headers_names = self.get_used_headers(with_key)

    keep_indexes = [self.HEADERS.index(n) for n in headers_names]
    return [self.HEADERS_DEFAULT[i] for i in keep_indexes]

  def get_all_names(self):
    return self.HEADERS

  def get_all_defaults(self):
    return self.HEADERS_DEFAULT

  def get_unused(self):
    return self.UNUSED

  def get_target_name(self):
    return self.TARGET_NAME

  def _make_base_features(self):
    continuous = {
        key_name: tfc.numeric_column(key_name)
        for key_name in self.numerics_names.keys()
    }
    categorical = {
        key_name: tfc.categorical_column_with_vocabulary_list(
            key=key_name,
            vocabulary_list=voc)
        for key_name, voc in self.categorical_names.items()
    }    
    return continuous, categorical

  def get_base_features(self):
    return self.continous, self.categorical

  def _prepare_for_crossing(self, key_name, num_bck, boundaries):
    key = None
    if key_name in self.continuous.keys():
      if boundaries is not None:
        key = tfc.bucketized_column(self.continuous[key_name], boundaries)
      else:
        key = tfc.categorical_column_with_identity(key_name, num_bck)
    elif key_name in self.categorical.keys():
      key = key_name
    else:
      key = key_name
    return key

  def _make_crossed(self):
    f_crossed_for_wide = []
    f_crossed_for_deep = []
    for to_cross in self.CROSSED:
      key = []
      bck_size = 1
      for (key, bck, bnd) in to_cross:
        keys.append(self._prepare_for_crossing(key, bck, bnd))
        bck_size *= bck

      t_crossed = tfc.crossed_column(keys, min(bck_size, 10000))
      t_dimension = int(bck_size**0.25)
      f_crossed_for_wide.append(t_crossed)
      f_crossed_for_deep.append(tfc.embedding_column(t_crossed, t_dimension))
    return f_crossed_for_wide, f_crossed_for_deep

  def get_wide_features(self):
    wide_features = self.categorical.values()
    if not self.ignore_crosses:
      wide_features += self.crossed_for_wide
    return wide_features  

  def get_deep_features(self, with_continuous=True):
    deep_features = [tfc.indicator_column(f) for f in self.categorical.values()]
    if with_continuous:
      deep_features += self.continuous.values()
    if not self.ignore_crosses:
      deep_features += self.crossed_for_deep
    return deep_features

#@title DNN: model.py

clvf = CLVFeatures(ignore_crosses=True)

def parse_csv(csv_row):
  columns = tf.decode_csv(csv_row, record_defaults = clvf.get_all_defaults())
  features = dict(zip(clvf.get_all_names(), columns))
  
  for column_name in clvf.get_unused():
    features.pop(column_name)

  target = features.pop(clvf.get_target_name())

  return features, target


def dataset_input_fn(data_folder, prefix=None, mode=None, params=None, count=None):
  shuffle = True if mode == tf.estimator.ModeKeys.TRAIN else False

  filenames = tf.matching_files('{}{}*.csv'.format(data_folder, prefix))

  dataset = tf.data.TextLineDataset(filenames).skip(1)
  dataset = dataset.map(parse_csv)
  if shuffle:
    dataset = dataset.shuffle(buffer_size=params.buffer_size)
  dataset = dataset.repeat(count=count)
  dataset = dataset.batch(params.batch_size)

  iterator = tf.compat.v1.data.make_one_shot_iterator(dataset) 
  
  features, target = iterator.get_next()

  return features, target

def read_train(data_folder, params):
  return dataset_input_fn(
      data_folder=data_folder,
      prefix='train',
      params=params,
      mode=tf.estimator.ModeKeys.TRAIN)


def read_eval(data_folder, params):
  return dataset_input_fn(data_folder=data_folder,
                          prefix='eval',
                          params=params)


def read_test(data_folder, params):
  return dataset_input_fn(data_folder=data_folder,
                          prefix='test',
                          params=params,
                          count=1)

def rmse_evaluator(labels, predictions):
  pred_values = predictions['predictions']
  return {'rmse': tf.metrics.root_mean_squared_error(labels, pred_values)} 

def get_learning_rate(params):
  global_step = tf.train.get_global_step()
  learning_rate = tf.train.exponential_decay(                    
      learning_rate = params.learning_rate,
      global_step = global_step,                                                                
      decay_steps = params.checkpoint_steps,
      decay_rate = params.learning_decay_rate,
      staircase = True)
  return learning_rate

def get_optimizer(params):
  optimizer = tf.train.ProximalAdagradOptimizer(
      learning_rate = get_learning_rate(params),                                                
      l1_regularization_strength = params.l1_regularization,
      l2_regularization_strength = params.l2_regularization)
  return optimizer

def get_estimator(config, params, model_dir):   
  estimator = tf.estimator.DNNRegressor(
      feature_columns=clvf.get_deep_features(),
      hidden_units=params.hidden_units,
      config=config,
      model_dir=model_dir,
      optimizer=lambda: get_optimizer(params),
      batch_norm=True,
      dropout=params.dropout)
  
  estimator = tf.contrib.estimator.add_metrics(estimator, rmse_evaluator)       
  return estimator

#@title DNN: task.py (Hyperparameter + parser.args)

TRAIN_SIZE = 100000      #length of trainset is 883, but this param is given
NUM_EPOCHS = 70
BATCH_SIZE = 5
NUM_EVAL = 20

LEARNING_DECAY_RATE = 0.7
HIDDEN_UNITS = '128 64 32 16'
LEARNING_RATE = 0.00135
L1_REGULARIZATION = 0.0216647
L2_REGULARIZATION = 0.0673949
DROPOUT = 0.899732
SHUFFLE_BUFFER_SIZE = 10000

job_dir = '/content/model_checkpoint/'
data_src = '/content/DNNs_for_CLVs/'
ignore_crosses = False #default
learning_rate_decay = True

hypertune = False #adds numbers to ouputpath when turned on 'True'
resume = False #default (takes old savings for start if turned on 'True -> you may turn off hypertune then?)


def csv_serving_input_fn():
  clvf = CLVFeatures(ignore_crosses=True)
  used_headers = clvf.get_used_headers(with_key=True, with_target=False)
  default_values = clvf.get_defaults(used_headers)

  rows_string_tensor = tf.placeholder(dtype=tf.string, shape=[None],      #not compatible with egaer execution and tf.function?
                                      name='csv_rows')
  receiver_tensor = {'csv_rows': rows_string_tensor}

  row_columns = tf.expand_dims(rows_string_tensor, -1)
  columns = tf.decode_csv(row_columns, record_defaults=default_values)

  features = dict(zip(used_headers, columns))

  return tf.estimator.export.ServingInputReceiver(features, receiver_tensor)

#@title DNN: task.py (main execution)

tf.logging.set_verbosity(tf.compat.v1.logging.INFO)

if hypertune:
  config = json.loads(os.environ.get('TF_CONFIG', '{}'))
  trial = config.get('task', {}).get('trial', '')
  model_dir = os.path.join(job_dir, trial)
else:
  model_dir = job_dir

data_folder = '{}'.format(data_src)

train_steps = (TRAIN_SIZE/BATCH_SIZE) * NUM_EPOCHS
checkpoint_steps = int((TRAIN_SIZE/BATCH_SIZE) * (
      NUM_EPOCHS/NUM_EVAL))

config = tf.estimator.RunConfig(
    save_checkpoints_steps=checkpoint_steps
)

hidden_units = [int(n) for n in HIDDEN_UNITS.split()]

params = tf.contrib.training.HParams(
    num_epochs = NUM_EPOCHS,
    train_steps = train_steps,
    batch_size = BATCH_SIZE,
    hidden_units = hidden_units,
    learning_rate = LEARNING_RATE,
    ignore_crosses = ignore_crosses,
    buffer_size = SHUFFLE_BUFFER_SIZE,
    learning_rate_decay = learning_rate_decay,
    learning_decay_rate = LEARNING_DECAY_RATE,
    l1_regularization = L1_REGULARIZATION,
    l2_regularization = L2_REGULARIZATION,
    dropout= DROPOUT,
    checkpoint_steps = checkpoint_steps)

estimator = None

estimator = get_estimator(config=config,
                          params=params,
                          model_dir=model_dir)

train_spec = tf.estimator.TrainSpec(
    input_fn=lambda: read_train(data_folder, params),
    max_steps=train_steps)

eval_spec = tf.estimator.EvalSpec(
    input_fn=lambda: read_eval(data_folder, params),
    exporters=[
        tf.estimator.LatestExporter(
            name='estimate',
            serving_input_receiver_fn=csv_serving_input_fn,
            exports_to_keep=1,
            as_text=True
        )
    ],
    steps=1000,
    throttle_secs=1,
    start_delay_secs=1
)

if not resume:
    print('Removing previous trained model...')
    shutil.rmtree(model_dir, ignore_errors=True)
else:
    print('Resuming training...')

tf.estimator.train_and_evaluate(estimator, train_spec, eval_spec)
        
estimator.evaluate(lambda: read_test(data_folder, params), name="Test Set")

{'average_loss': 17710520.0,
 'global_step': 12362,
 'label/mean': 3189.7307,
 'loss': 87154410.0,
 'prediction/mean': 0.2885886,
 'rmse': 4208.3867}

#@title DNN: context.py

class CLVFeatures(object):

  HEADERS = ['customer_id', 'monetary_dnn', 'monetary_btyd', 'frequency_dnn',
             'frequency_btyd', 'recency', 'T', 'time_between',
             'avg_basket_value', 'avg_basket_size', 'cnt_returns',
             'has_returned', 'frequency_btyd_clipped', 'monetary_btyd_clipped',
             'target_monetary_clipped', 'target_monetary']
  HEADERS_DEFAULT = [[''], [0.0], [0.0], [0],
                     [0], [0], [0], [0.0],
                     [0.0], [0.0], [0],
                     [-1], [0], [0.0],
                     [0.0], [0.0]]
  NUMERICS = {
      'monetary_dnn': [],
      'recency': [],
      'frequency_dnn': [],
      'T': [],
      'time_between': [],
      'avg_basket_value': [],
      'avg_basket_size': [],
      'cnt_returns': []}
  CATEGORICALS_W_LIST = {
      'has_returned': [0, 1]}
  CROSSED = []
  KEY = 'customer_id'
  UNUSED = [KEY, 'monetary_btyd', 'frequency_btyd', 'frequency_btyd_clipped',
            'monetary_btyd_clipped', 'target_monetary_clipped']
  TARGET_NAME = 'target_monetary'

  def __init__(self, ignore_crosses=False): #, is_dnn=None

    #if not is_dnn:
    #  return

    self.ignore_crosses = ignore_crosses
    (self.headers, self.numerics_names, self.categorical_names) = self._keep_used()
    self.continuous, self.categorical = self._make_base_features()

    if not self.ignore_crosses:
      self.crossed_for_wide, self.crossed_for_deep = self._make_crossed()
  
  def _keep_used(self):
    headers = [h for h in self.HEADERS if h not in self.UNUSED]
    numerics_names = {
        k: v for k, v in iteritems(self.NUMERICS)
        if (k not in self.UNUSED) and (k != self.TARGET_NAME)
    }
    categorical_names = {
        k: v for k, v in iteritems(self.CATEGORICALS_W_LIST)
        if k not in self.UNUSED    
    }                          
    return headers, numerics_names, categorical_names
  
  def get_key(self):
    return self.KEY
  
  def get_used_headers(self, with_key=False, with_target=False):
    used_headers = [h for h in self.headers if h != self.TARGET_NAME]

    if with_key:
      used_headers.insert(0, self.KEY)
    if with_target:
      used_headers.append(self.TARGET_NAME)

    return used_headers

  def get_defaults(self, headers_names=None, with_key=False):
    if headers_names is None:
      headers_names = self.get_used_headers(with_key)

    keep_indexes = [self.HEADERS.index(n) for n in headers_names]
    return [self.HEADERS_DEFAULT[i] for i in keep_indexes]

  def get_all_names(self):
    return self.HEADERS

  def get_all_defaults(self):
    return self.HEADERS_DEFAULT

  def get_unused(self):
    return self.UNUSED

  def get_target_name(self):
    return self.TARGET_NAME

  def _make_base_features(self):
    continuous = {
        key_name: tfc.numeric_column(key_name)
        for key_name in self.numerics_names.keys()
    }
    categorical = {
        key_name: tfc.categorical_column_with_vocabulary_list(
            key=key_name,
            vocabulary_list=voc)
        for key_name, voc in self.categorical_names.items()
    }    
    return continuous, categorical

  def get_base_features(self):
    return self.continous, self.categorical

  def _prepare_for_crossing(self, key_name, num_bck, boundaries):
    key = None
    if key_name in self.continuous.keys():
      if boundaries is not None:
        key = tfc.bucketized_column(self.continuous[key_name], boundaries)
      else:
        key = tfc.categorical_column_with_identity(key_name, num_bck)
    elif key_name in self.categorical.keys():
      key = key_name
    else:
      key = key_name
    return key

  def _make_crossed(self):
    f_crossed_for_wide = []
    f_crossed_for_deep = []
    for to_cross in self.CROSSED:
      key = []
      bck_size = 1
      for (key, bck, bnd) in to_cross:
        keys.append(self._prepare_for_crossing(key, bck, bnd))
        bck_size *= bck

      t_crossed = tfc.crossed_column(keys, min(bck_size, 10000))
      t_dimension = int(bck_size**0.25)
      f_crossed_for_wide.append(t_crossed)
      f_crossed_for_deep.append(tfc.embedding_column(t_crossed, t_dimension))
    return f_crossed_for_wide, f_crossed_for_deep

  def get_wide_features(self):
    wide_features = self.categorical.values()
    if not self.ignore_crosses:
      wide_features += self.crossed_for_wide
    return wide_features  

  def get_deep_features(self, with_continuous=True):
    deep_features = [tfc.indicator_column(f) for f in self.categorical.values()]
    if with_continuous:
      deep_features += self.continuous.values()
    if not self.ignore_crosses:
      deep_features += self.crossed_for_deep
    return deep_features

#@title DNN: context.py

class CLVFeatures(object):

  HEADERS = ['customer_id', 'monetary_dnn', 'monetary_btyd', 'frequency_dnn',
             'frequency_btyd', 'recency', 'T', 'time_between',
             'avg_basket_value', 'avg_basket_size', 'cnt_returns',
             'has_returned', 'frequency_btyd_clipped', 'monetary_btyd_clipped',
             'target_monetary_clipped', 'target_monetary']
  HEADERS_DEFAULT = [[''], [0.0], [0.0], [0],
                     [0], [0], [0], [0.0],
                     [0.0], [0.0], [0],
                     [-1], [0], [0.0],
                     [0.0], [0.0]]
  NUMERICS = {
      'monetary_dnn': [],
      'recency': [],
      'frequency_dnn': [],
      'T': [],
      'time_between': [],
      'avg_basket_value': [],
      'avg_basket_size': [],
      'cnt_returns': []}
  CATEGORICALS_W_LIST = {
      'has_returned': [0, 1]}
  CROSSED = []
  KEY = 'customer_id'
  UNUSED = [KEY, 'monetary_btyd', 'frequency_btyd', 'frequency_btyd_clipped',
            'monetary_btyd_clipped', 'target_monetary_clipped']
  TARGET_NAME = 'target_monetary'

  def __init__(self, ignore_crosses=False): #, is_dnn=None

    #if not is_dnn:
    #  return

    self.ignore_crosses = ignore_crosses
    (self.headers, self.numerics_names, self.categorical_names) = self._keep_used()
    self.continuous, self.categorical = self._make_base_features()

    if not self.ignore_crosses:
      self.crossed_for_wide, self.crossed_for_deep = self._make_crossed()
  
  def _keep_used(self):
    headers = [h for h in self.HEADERS if h not in self.UNUSED]
    numerics_names = {
        k: v for k, v in iteritems(self.NUMERICS)
        if (k not in self.UNUSED) and (k != self.TARGET_NAME)
    }
    categorical_names = {
        k: v for k, v in iteritems(self.CATEGORICALS_W_LIST)
        if k not in self.UNUSED    
    }                          
    return headers, numerics_names, categorical_names
  
  def get_key(self):
    return self.KEY
  
  def get_used_headers(self, with_key=False, with_target=False):
    used_headers = [h for h in self.headers if h != self.TARGET_NAME]

    if with_key:
      used_headers.insert(0, self.KEY)
    if with_target:
      used_headers.append(self.TARGET_NAME)

    return used_headers

  def get_defaults(self, headers_names=None, with_key=False):
    if headers_names is None:
      headers_names = self.get_used_headers(with_key)

    keep_indexes = [self.HEADERS.index(n) for n in headers_names]
    return [self.HEADERS_DEFAULT[i] for i in keep_indexes]

  def get_all_names(self):
    return self.HEADERS

  def get_all_defaults(self):
    return self.HEADERS_DEFAULT

  def get_unused(self):
    return self.UNUSED

  def get_target_name(self):
    return self.TARGET_NAME

  def _make_base_features(self):
    continuous = {
        key_name: tfc.numeric_column(key_name)
        for key_name in self.numerics_names.keys()
    }
    categorical = {
        key_name: tfc.categorical_column_with_vocabulary_list(
            key=key_name,
            vocabulary_list=voc)
        for key_name, voc in self.categorical_names.items()
    }    
    return continuous, categorical

  def get_base_features(self):
    return self.continous, self.categorical

  def _prepare_for_crossing(self, key_name, num_bck, boundaries):
    key = None
    if key_name in self.continuous.keys():
      if boundaries is not None:
        key = tfc.bucketized_column(self.continuous[key_name], boundaries)
      else:
        key = tfc.categorical_column_with_identity(key_name, num_bck)
    elif key_name in self.categorical.keys():
      key = key_name
    else:
      key = key_name
    return key

  def _make_crossed(self):
    f_crossed_for_wide = []
    f_crossed_for_deep = []
    for to_cross in self.CROSSED:
      key = []
      bck_size = 1
      for (key, bck, bnd) in to_cross:
        keys.append(self._prepare_for_crossing(key, bck, bnd))
        bck_size *= bck

      t_crossed = tfc.crossed_column(keys, min(bck_size, 10000))
      t_dimension = int(bck_size**0.25)
      f_crossed_for_wide.append(t_crossed)
      f_crossed_for_deep.append(tfc.embedding_column(t_crossed, t_dimension))
    return f_crossed_for_wide, f_crossed_for_deep

  def get_wide_features(self):
    wide_features = self.categorical.values()
    if not self.ignore_crosses:
      wide_features += self.crossed_for_wide
    return wide_features  

  def get_deep_features(self, with_continuous=True):
    deep_features = [tfc.indicator_column(f) for f in self.categorical.values()]
    if with_continuous:
      deep_features += self.continuous.values()
    if not self.ignore_crosses:
      deep_features += self.crossed_for_deep
    return deep_features

#@title DNN: model.py

clvf = CLVFeatures(ignore_crosses=True)

def parse_csv(csv_row):
  columns = tf.io.decode_csv(csv_row, record_defaults = clvf.get_all_defaults())
  features = dict(zip(clvf.get_all_names(), columns))
  
  for column_name in clvf.get_unused():
    features.pop(column_name)

  target = features.pop(clvf.get_target_name())

  return features, target


def dataset_input_fn(data_folder, prefix=None, mode=None, count=None):
  shuffle = True if mode == tf.estimator.ModeKeys.TRAIN else False

  filenames = tf.io.matching_files('{}{}*.csv'.format(data_folder, prefix))
  dataset = tf.data.TextLineDataset(filenames).skip(1)
  dataset = dataset.map(parse_csv)
  if shuffle:
    dataset = dataset.shuffle(buffer_size=SHUFFLE_BUFFER_SIZE)
  dataset = dataset.repeat(count=count)
  dataset = dataset.batch(BATCH_SIZE)

  iterator = tf.compat.v1.data.make_one_shot_iterator(dataset)   #tf.compat.v1.data.make_one_shot_iterator(dataset)/#tf.compat.v1.data.make_initializable_iterator(dataset)
  
  features, target = iterator.get_next()

  return features, target

def read_train(data_folder):
  return dataset_input_fn(
      data_folder=data_folder,
      prefix='train',
      mode=tf.estimator.ModeKeys.TRAIN)


def read_eval(data_folder):
  return dataset_input_fn(data_folder=data_folder,
                          prefix='eval')


def read_test(data_folder):
  return dataset_input_fn(data_folder=data_folder,
                          prefix='test',
                          count=1)

def rmse_evaluator(labels, predictions):
  pred_values = predictions['predictions']
  return {'rmse': tf.compat.v1.metrics.root_mean_squared_error(labels, pred_values)} #tf.cast(labels, tf.float32)

def get_learning_rate():
  global_step = tf.compat.v1.train.get_global_step()
  learning_rate = tf.compat.v1.train.exponential_decay(                    
      learning_rate = LEARNING_RATE,
      global_step = global_step,                                                                
      decay_steps = checkpoint_steps,
      decay_rate = LEARNING_DECAY_RATE,
      staircase = True)
  return learning_rate

def get_optimizer():
  optimizer = tfa.optimizers.ProximalAdagrad(
      learning_rate = get_learning_rate(),                                                
      l1_regularization_strength = L1_REGULARIZATION,
      l2_regularization_strength = L2_REGULARIZATION)
  return optimizer

def get_estimator(config, model_dir):    #estimator_name
  estimator = tf.estimator.DNNRegressor(
      feature_columns=clvf.get_deep_features(),
      hidden_units=hidden_units,
      config=config,
      model_dir=model_dir,
      optimizer=lambda: get_optimizer(),
      batch_norm=True,
      dropout=DROPOUT)
  
  estimator = tf.estimator.add_metrics(estimator, rmse_evaluator)       
  return estimator

#@title DNN: task.py (Hyperparameter + parser.args)

TRAIN_SIZE = 100000      #actually the length of the train set is 883 (googles default: 100000)
NUM_EPOCHS = 70
BATCH_SIZE = 5
NUM_EVAL = 20

LEARNING_DECAY_RATE = 0.7
HIDDEN_UNITS = '128 64 32 16'
LEARNING_RATE = 0.00135
L1_REGULARIZATION = 0.0216647
L2_REGULARIZATION = 0.0673949
DROPOUT = 0.899732
SHUFFLE_BUFFER_SIZE = 10000

hypertune = False #adds numbers to ouputpath when turned on 'True'
job_dir = ('/content/model_checkpoint')
data_src = ('/content/DNNs_for_CLVs')
ignore_crosses = False #default
learning_rate_decay = True
resume = False #default


def csv_serving_input_fn():
  clvf = CLVFeatures(ignore_crosses=True)
  used_headers = clvf.get_used_headers(with_key=True, with_target=False)
  default_values = clvf.get_defaults(used_headers)

  rows_string_tensor = tf.compat.v1.placeholder(dtype=tf.string, shape=[None],      #not compatible with egaer execution and tf.function?
                                      name='csv_rows')
  receiver_tensor = {'csv_rows': rows_string_tensor}

  row_columns = tf.compat.v1.expand_dims(rows_string_tensor, -1)
  columns = tf.io.decode_csv(row_columns, record_defaults=default_values)

  features = dict(zip(used_headers, columns))

  return tf.estimator.export.ServingInputReceiver(features, receiver_tensor)

#@title DNN: task.py (main execution)

tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.INFO)

if hypertune:
  config = json.loads(os.environ.get('TF_CONFIG', '{}'))
  trial = config.get('task', {}).get('trial', '')
  model_dir = os.path.join(job_dir, trial)
else:
  model_dir = job_dir

data_folder = '{}/'.format(data_src)

train_steps = (TRAIN_SIZE/BATCH_SIZE) * NUM_EPOCHS
checkpoint_steps = int((TRAIN_SIZE/BATCH_SIZE) * (
      NUM_EPOCHS/NUM_EVAL))

config = tf.estimator.RunConfig(
    save_checkpoints_steps=checkpoint_steps
)

hidden_units = [int(n) for n in HIDDEN_UNITS.split()]

estimator = None

estimator = get_estimator(config=config,
                          model_dir=model_dir)

train_spec = tf.estimator.TrainSpec(
    input_fn=lambda: read_train(data_folder),   #but the input IS a tuple!
    max_steps=train_steps)

eval_spec = tf.estimator.EvalSpec(
    input_fn=lambda: read_eval(data_folder),
    exporters=[
        tf.estimator.LatestExporter(
            name='estimate',
            serving_input_receiver_fn=csv_serving_input_fn,
            exports_to_keep=1,
            as_text=True
        )
    ],
    steps=1000,
    throttle_secs=1,
    start_delay_secs=1
)

if not resume:
    print('Removing previous trained model...')
    shutil.rmtree(model_dir, ignore_errors=True)
else:
    print('Resuming training...')

tf.estimator.train_and_evaluate(estimator, train_spec, eval_spec)
        
estimator.evaluate(lambda: read_test(data_folder), name="Test Set")

{'average_loss': 17708200.0,
 'global_step': 12362,
 'label/mean': 3189.7307,
 'loss': 20264198.0,
 'prediction/mean': 0.66893655,
 'rmse': 4208.1113}