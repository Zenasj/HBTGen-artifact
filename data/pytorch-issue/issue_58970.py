import torch
import tempfile

a = torch.randn(10, dtype=torch.complex64)

# a.real and a.imag share the same data with a, but they have a real dtype, not complex
ar = a.real
ai = a.imag

a_s = a.storage()
ar_s = ar.storage()
ai_s = ai.storage()

print(f'type(a_s): {type(a_s)}')
print(f'type(ar_s): {type(ar_s)}')
print(f'type(ai_s): {type(ai_s)}')

assert a_s._cdata == ar_s._cdata
assert a_s._cdata == ai_s._cdata

with tempfile.NamedTemporaryFile() as f:
    torch.save([a, ar, ai], f)
    f.seek(0)
    a_l, ar_l, ai_l = torch.load(f)