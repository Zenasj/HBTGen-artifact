class Mod(Function):
    @classmethod
    def eval(cls, p, q):
        # Never simplify Mod expressions.
        return None