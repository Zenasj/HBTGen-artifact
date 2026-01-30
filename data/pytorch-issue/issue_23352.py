import torch
import torch.nn as nn

EMBEDDING_DIM = 6
HIDDEN_DIM = 6

token_embedding = Embedding(num_embeddings=vocab.get_vocab_size('tokens'),
embedding_dim=EMBEDDING_DIM)
word_embeddings = BasicTextFieldEmbedder({"tokens": token_embedding})
lstm = PytorchSeq2SeqWrapper(torch.nn.LSTM(EMBEDDING_DIM, HIDDEN_DIM, batch_first=True))
model = LstmTagger(word_embeddings, lstm, vocab)

class MLP(nn.Module):
    def __init__(self, dims: List[int], dropout: float = 0.):
        super().__init__()
        layers = []
        for in_dim, dim in zip([-1] + dims, dims):
            layers.extend([
                nn.Linear(in_dim, dim),
                ReLU(),
                nn.LayerNorm(dim),
                nn.Dropout(dropout),
            ])
        self.layers = nn.Sequential(*layers)

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        return self.layers(input)

mlp1 = MLP([4, 5, 6])
mlp2 = MLP([4, 5, 6])

mlp1(torch.rand(2, 4))  # finalizes mlp1 as expecting input dimension 4
mlp2(torch.rand(2, 5))  # finalizes mlp2 as expecting input dimension 5

class Model(nn.Module):
    def __init__(
        self,
        embedding = WordEmbedding(),
        representation = BiLSTM(),
        decoder = MLP([4, 5, 6]),
    ):
        self.embedding = embedding
        self.representation = representation
        self.decoder = decoder
    
    def forward(self, tokens):
        embedded = self.embedding(tokens)
        representation = self.representation(embedded)
        return self.decoder(representation)
        
model1 = Model()
model2 = Model(representation=DocNN())
model3 = Model(
    embedding=WordEmbedding(embedding_dim=200, vocab_size=10000),
    decoder=MLP([2, 50, 10]),
)

from torch import nn

# LazyModuleMeta re-implements the type construction semantics for objects to allow
# a slight variant on syntax. Essentially anything with this metaclass can optionally
# execute a single yield statement during its constructor (normal constructors also work fine).
# If it does yield during construction, then a __lazy_init__ function is populated;
# any code occurring before yield in the constuctor will be called as normal during object creation,
# and any code after yield will instead be deferred to the first call of __lazy_init__.

class LazyInitMeta(type):
    def __call__(cls, *args, **kwargs):
        if hasattr(cls, '__new__'):
            obj = cls.__new__(cls, *args, **kwargs)
        else:
            obj = object.__new__(cls)
        
        def initialize(obj):
            res = obj.__init__(*args, **kwargs)
            if isinstance(res, types.GeneratorType):
                next(res, None)
                def lazy_init(call_args):
                    try:
                        res.send(call_args)
                    except StopIteration:
                        pass
                    finally:
                        obj.__lazy_init__ = None
                obj.__lazy_init__ = lazy_init
            else:
                obj.__lazy_init__ = None
            
        if isinstance(obj, cls):
            initialize(obj)
        return obj
        
# Here's a Lazy nn.Module implementation using LazyInitMeta, calling __lazy_init__ before the first
# forward pass.
    
class LazyModule(nn.Module, metaclass=LazyInitMeta):
    def __init__(self):
        nn.Module.__init__(self)
    
    def __call__(self, *args, **kwargs):
        if self.__lazy_init__:
            self.__lazy_init__(call_args=(args, kwargs))
        return nn.Module.__call__(self, *args, **kwargs)
        
# Optionally lazy Linear module, based on the current torch implementation of nn.Linear.

class Linear(nn.Linear, LazyModule):
    def __init__(self, in_features, out_features, bias=True):
        LazyModule.__init__(self)
        self.in_features = in_features
        self.out_features = out_features
        if bias:
            self.bias = nn.Parameter(torch.Tensor(out_features))
        else:
            self.register_parameter('bias', None)
            
        if in_features == -1:
            self.register_parameter('weight', None)
            ([input], _) = yield  # lazy init remainder
            in_features = self.in_features = input.size()[-1]
            
        self.weight = nn.Parameter(torch.Tensor(out_features, in_features))
        self.reset_parameters()

model = Model(...)

# Should error out in `model.parameters()` if you execute this line, as it encounters UninitializedParameter
optimizer = optim.SGD(model.parameters(), lr = 0.01)  # No error

# finalize by sending an example input
inp = torch.randn(10, 20)
output = model(inp)

# No error
optimizer = optim.SGD(model.parameters(), lr = 0.01)