w.join(15.0) # timeout in 15s
if w.is_alive():
    w.terminate()

while True:
    self._workers[wid].join(15.0) # timeout in 15s
    if self._workers[wid].is_alive():
        self._workers_status[wid] = True
        self._shutdown_worker(wid)
    else:
        break