import torch as t
import itertools

Id_max = 10
I,J = 13,14

index = t.randint(0,Id_max,(1,J))
src = t.rand(I,J)

# ground truth
y_correct = t.zeros(I,Id_max)
for i,j in itertools.product(range(I),range(J),):
    j_idx = index[0,j]
    y_correct[i,j_idx] = y_correct[i,j_idx] + src[i,j]

# incorrect answer with broadcasting
y1 = t.zeros(
        I,Id_max)
y1.scatter_add_(
    src=src,
    index=index,
    dim=1,
)
print(t.max(t.abs(y_correct-y1)))

# correct answer without broadcasting
index = index.expand(I,-1)
y2 = t.zeros(
        I,Id_max)
y2.scatter_add_(
    src=src,
    index=index,
    dim=1,
)
print(t.max(t.abs(y_correct-y2)))