for i in range(self.num_workers):
                print(54.1)
                index_queue = multiprocessing.Queue()
                print(54.2)
                index_queue.cancel_join_thread()
                print(54.3)
                w = multiprocessing.Process(
                    target=_worker_loop,
                    args=(self.dataset, index_queue,
                          self.worker_result_queue, self.done_event,
                          self.collate_fn, base_seed + i,
                          self.worker_init_fn, i))
                print(54.4)
                w.daemon = True
                print(54.5)
                # NB: Process.start() actually take some time as it needs to
                #     start a process and pass the arguments over via a pipe.
                #     Therefore, we only add a worker to self.workers list after
                #     it started, so that we do not call .join() if program dies
                #     before it starts, and __del__ tries to join but will get:
                #     AssertionError: can only join a started process.
                w.start()
                print(54.6)
                self.index_queues.append(index_queue)
                print(54.7)
                self.workers.append(w)