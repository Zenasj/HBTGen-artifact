import tensorflow as tf
from tf.python.profiler import profiler_client

resolver = tf.distribute.cluster_resolver.TPUClusterResolver(tpu="local")
tf.tpu.experimental.initialize_tpu_system(resolver)
strategy = tf.distribute.TPUStrategy(resolver)

class Monitor(object):
    def init(self, service_addr, duration_ms):
        self.service_addr = service_addr
        self.duration_ms = duration_ms
        self._stop = True
        self.client = profiler_client

    def _loop(self):
       while not self._stop:
            time.sleep(0.5)
            try:
                self.client.monitor(self.service_addr, duration_ms=self.duration_ms, level=1)
            except Exception as e:
                print(e)
                time.sleep(1)

    def start(self):
        if self._stop:
            self._thread = threading.Thread(target=self._loop, daemon=True)
            self._stop = False
            self._thread.start()

    def stop(self):
        if not self._stop:
            self._stop = True
            self._thread.join()

tf.profiler.experimental.server.start(8466)
tpu_monitor = Monitor("grpc://localhost:8466", 2000)
tpu_monitor.start()
train_model()
tpu_monitor.stop()

resolver = tf.distribute.cluster_resolver.TPUClusterResolver(tpu="local")