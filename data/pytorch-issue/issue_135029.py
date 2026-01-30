diff
- setattr(dc, name, apply(getattr(dc, name)))
+ object.__setattr__(dc, name, apply(getattr(dc, name)))