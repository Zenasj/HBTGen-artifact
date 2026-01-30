def ___make_guard_fn():
    def guard(L):
        if not (x[0].a < x[1].a * (3 - x[2].a)):
            return False
        if not (a.b.c[0].d.e + a.b.c[1].d.e * a.b.c[2].d.e > 0):
            return False
        if not (f(m.n[0], '0').x.y.z * f(m.n[0], '1').x.y.z * f(m.n[0], '2').x.y.z < 512):
            return False
        if not (self.g(a, b).k + (1 - self.g(a, b).k) <= m[0].a + self.g(a, b).k):
            return False
        return True
    return guard