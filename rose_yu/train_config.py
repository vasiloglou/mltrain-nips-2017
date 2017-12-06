class TrainConfig(object):
  """Tiny config, for testing."""
  init_scale = 0.1
  learning_rate = 1e-2
  decay_rate = 0.8
  max_grad_norm = 10
  hidden_size = 16 # dim of h
  num_layers = 2
  inp_steps = 12 
  horizon = 1
  num_lags = 2 # num prev hiddens
  num_orders = 2 # tensor prod order
  rank_vals= [2]
  num_freq = 2
  training_steps = int(5e3)
  keep_prob = 1.0 # dropout
  sample_prob = 0.0 # sample ground true
  batch_size = 50
