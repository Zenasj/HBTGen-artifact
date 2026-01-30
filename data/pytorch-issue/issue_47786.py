import fileinput
items = ''.join(fileinput.input()).split('\n\n')
print(''.join(sorted(f'\n{item.strip()}\n\n' for item in items)), end='')