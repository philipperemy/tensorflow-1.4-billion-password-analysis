from constants import EDIT_DISTANCE_FILENAME


def stream_from_file():
    with open(EDIT_DISTANCE_FILENAME, 'rb') as r:
        for l in r.readlines():
            ed, x, y = l.decode('utf8').strip().split(' ||| ')
            yield x.strip(), y.strip()


class LazyDataLoader:
    def __init__(self):
        self.stream = stream_from_file()

    def next(self):
        try:
            return next(self.stream)
        except:
            self.stream = stream_from_file()
            return self.next()

    def statistics(self):
        max_len_value_x = 0
        max_len_value_y = 0
        num_lines = 0
        self.stream = stream_from_file()
        for x, y in self.stream:
            max_len_value_x = max(max_len_value_x, len(x))
            max_len_value_y = max(max_len_value_y, len(y))
            num_lines += 1

        print('max_len_value_x =', max_len_value_x)
        print('max_len_value_y =', max_len_value_y)
        print('num_lines =', num_lines)
        return max_len_value_x, max_len_value_y, num_lines


if __name__ == '__main__':
    # how to use it.
    ldl = LazyDataLoader()
    print(ldl.statistics())
    while True:
        print(ldl.next())
