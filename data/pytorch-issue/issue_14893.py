class Base(object):
    def __init__(self):
        self.alpha = ["a", "b", "c", "d"]

    def __getattr__(self, attr):
        if attr in self.alpha:
            for x in self.alpha:
                yield getattr(x, attr)


class Sub(Base):
    pass


obj = Sub()
getattr(obj, "__reduce_ex__")(2)  # this is basically what pickle was doing

class Sub(Base):
    def __getattr__(self, attr):
        # saveguard against infinite recursion when 'alpha' DNE
        if 'alpha' not in vars(self):
            raise AttributeError
        if attr in self.alpha:
            # RETURN a generator
            return (getattr(x, attr) for x in self.alpha)
        else:
            raise AttributeError

obj = Sub()
getattr(obj, "__reduce_ex__")(2)