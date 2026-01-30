def worker_init_func():
  workerInfo.dataset = Dataset(...)

def worker_init_func():
  dataset = workerInfo.dataset
  dataset.dataset.value = "foo"