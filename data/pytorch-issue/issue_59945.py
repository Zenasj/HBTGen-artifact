b = a.conj()
a[:10] = -99  # does this modify b now?? (it shouldn't!)