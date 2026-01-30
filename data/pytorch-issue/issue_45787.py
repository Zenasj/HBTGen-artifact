import torch

with torch.cuda.stream(side_stream):
    # loss.backward() implicitly synthesizes a one-element 1.0 tensor on side_stream
    # GraphRoot passes it to consumers, but consumers first sync on default stream, not side_stream.    
    loss.backward()

    # Internally to backward(), streaming-backward logic takes over, stuff executes on the same stream it ran on in forward,
    # and the side_stream context is irrelevant.  GraphRoot's interaction with its first consumer(s) is the spot where
    # the side_stream context causes a problem.

# implicit population is safe
with torch.cuda.stream(side_stream):
    loss.backward()

# explicit population in side stream then backward in side stream is safe
with torch.cuda.stream(side_stream):
    kickoff_grad = torch.ones_like(loss)
    loss.backward(gradient=kickoff_grad)

# explicit population in one stream then backward kickoff in another stream
# is NOT safe, even with this PR's diffs, but that unsafety is consistent with
# stream-semantics relationship of any pair of ops
kickoff_grad = torch.ones_like(loss)
with torch.cuda.stream(side_stream):
    loss.backward(gradient=kickoff_grad)

# Safe, as you'd expect for any pair of ops
kickoff_grad = torch.ones_like(loss)
side_stream.wait_stream(torch.cuda.current_stream())
with torch.cuda.stream(side_stream):
    loss.backward(gradient=kickoff_grad)