def show_me_data(d: Dataset) -> None:
    for x in d:
        print(x)

def __iter__(self):
        i = 0
        try:
            while True:
                v = self[i]
                yield v
                i += 1
        except IndexError:
            return