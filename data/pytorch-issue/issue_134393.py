import multiprocessing

def fn():
    tmp_list = ['a', 'b', 'c', 'd', 'a', 'c']
    print(set(tmp_list))

if __name__ == '__main__':
    p_list = []
    for i in range(10):
        p = multiprocessing.Process(target=fn)
        p_list.append(p)
        p.start()

    for p in p_list:
        p.join()

{'a', 'c', 'd', 'b'}
{'b', 'a', 'c', 'd'}
{'b', 'a', 'd', 'c'}
{'a', 'c', 'b', 'd'}
{'a', 'd', 'c', 'b'}
{'b', 'a', 'c', 'd'}
{'c', 'a', 'b', 'd'}
{'d', 'a', 'b', 'c'}
{'d', 'a', 'b', 'c'}
{'b', 'c', 'a', 'd'}