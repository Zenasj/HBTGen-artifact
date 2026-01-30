import threading
from multiprocessing import Process

import tensorflow as tf
from tensorflow.core.protobuf import config_pb2
from tensorflow.python import ops
from tensorflow.python.client.session import Session
from tensorflow.python.ops import collective_ops

cluster_spec = {
    "worker": [
        "localhost:14286",
        "localhost:14287"
    ]
}
inputs = [0.1, 1.1, 2.1, 3.1, 4.1, 5.1, 6.1, 7.1]
group_size = 4
group_key = 1
instance_key = 1
use_nccl = False
num_gpus_per_node = 2


def _configure(group_size):
    gpu_options = config_pb2.GPUOptions(
        visible_device_list='0,1',
        per_process_gpu_memory_fraction=0.7 / (group_size))
    experimental = config_pb2.ConfigProto.Experimental(collective_nccl=use_nccl)
    experimental.collective_group_leader = '/job:worker/replica:0/task:0'
    return config_pb2.ConfigProto(gpu_options=gpu_options, experimental=experimental)


class TFCluster(object):
    def __init__(self, cluster_spec):
        self._cluster_spec = cluster_spec
        self._num_worker = 0
        self._tf_servers = []

    def start(self):
        def server(job_name: str, task_index: int):
            server = tf.distribute.Server(self._cluster_spec,
                                          job_name=job_name,
                                          task_index=task_index,
                                          config=_configure(group_size))
            server.join()

        self._num_worker = len(cluster_spec.get("worker", []))
        assert self._num_worker >= 1
        for i in range(self._num_worker):
            self._tf_servers.append(Process(target=server,
                                            args=("worker", i), daemon=True))
        for proc in self._tf_servers:
            proc.start()

    def stop(self):
        for proc in self._tf_servers:
            proc.terminate()


def between_graph_test():
    def run_between_graph_clients(client_fn, cluster_spec, num_gpus, *args,
                                  **kwargs):
        threads = []
        for task_type in ['chief', 'worker']:
            for task_id in range(len(cluster_spec.get(task_type, []))):
                t = threading.Thread(
                    target=test_reduction,
                    args=(task_type, task_id, num_gpus) + args,
                    kwargs=kwargs)
                t.start()
                threads.append(t)
        for t in threads:
            t.join()

    def test_reduction(task_type,
                       task_id,
                       num_gpus):
        worker_device = "/job:%s/task:%d" % (task_type, task_id)
        master_target = "grpc://" + cluster_spec[task_type][0]
        with ops.Graph().as_default(), Session(target=master_target) as sess:
            collectives = []
            for i in range(num_gpus):
                with ops.device('/job:worker/task:0/device:CPU:0'):  # make sure all use the same variable
                    t = tf.Variable(inputs)
                with ops.device(worker_device + '/device:GPU:' + str(i)):
                    collectives.append(collective_ops.all_reduce(
                        t, group_size, group_key, instance_key, 'Add', 'Div'))
            run_options = config_pb2.RunOptions()
            run_options.experimental.collective_graph_key = 6
            sess.run(tf.compat.v1.global_variables_initializer())
            res_m = sess.run(collectives, options=run_options)
            print(res_m)

    run_between_graph_clients(
        test_reduction,
        cluster_spec,
        num_gpus_per_node)

# launch in-process clusters
cluster = TFCluster(cluster_spec)
cluster.start()

# run between graph execution
between_graph_test()
cluster.stop()