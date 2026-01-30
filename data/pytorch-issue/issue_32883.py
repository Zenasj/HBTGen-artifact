import torch

def test_two_threads_crash(self):
        import threading
        def torch_adds():
            with torch.autograd.profiler.profile() as prof:
                return torch.add(1,1)
            print(prof.key_averages().table(sort_by="self_cpu_time_total"))
        def torch_mul():
            with torch.autograd.profiler.profile() as p1:
                torch.mul(1,1)

        t1 = threading.Thread(target=torch_adds, args=())
        t2 = threading.Thread(target=torch_mul, args=())
        t1.start() ; t2.start()
        t1.join() ; t2.join()

def test_multithread_profiler(self):
        import threading
        def torch_mul():
            # yield
            import time ; time.sleep(0)
            return torch.mul(1,1)

        t2 = threading.Thread(target=torch_mul, args=())
        t2.start()
        with torch.autograd.profiler.profile() as prof:
            import time ; time.sleep(0) # yield
            torch.add(1,1)

        t2.join()
        print(prof.key_averages().table(sort_by="self_cpu_time_total"))