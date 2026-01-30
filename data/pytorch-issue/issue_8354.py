import torch

def forward(self,x):
        x0 = x.clone()
        torch._C._cuda_setStream(self.stream1._cdata)
        y0 = self.fc1(x0)
        self.event1.record(stream = torch.cuda.current_stream())
        
        torch._C._cuda_setStream(self.stream2._cdata)
        y1 = self.fc2(x)
        self.event2.record(stream = torch.cuda.current_stream())
        self.stream2.wait_event(self.event1)
        return y0 + y1

def backward_epilogue(self):
            torch.cuda.current_stream().wait_stream(self.reduction_stream)

if not self.callback_queued:
                                Variable._execution_engine.queue_callback(backward_epilogue)
                                self.callback_queued = True

torch.cuda.stashed_current_stream.wait_stream(self.reduction_stream)