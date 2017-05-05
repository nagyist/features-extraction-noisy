




def progress(count, total, status='', pycharm_fix=True):
    import sys
    bar_len = 60
    filled_len = int(round(bar_len * count / float(total)))
    percents = round(100.0 * count / float(total), 1)
    bar = '=' * filled_len + '-' * (bar_len - filled_len)
    if pycharm_fix:
        sys.stdout.write('\r[%s] %s%s %s' % (bar, percents, '%', status))    # works on pycharm
    else:
        sys.stdout.write('[%s] %s%s %s\r' % (bar, percents, '%', status))  # better for terminal, not working on pycharm
    sys.stdout.flush()




def range_from_skip_max_tot(skip, max_element, tot_element):
    n = max_element
    if max_element < 0 or max_element > tot_element-skip:
        n = tot_element - skip
    return range(skip, n+skip)

def subsequence_index(skip, n_element, tot_element):
    """
    Usage example:
        f, l, n = skip_n_tot(skip, n, len(lst))
        for elem in lst[f:l]:
            foo(elem)

        for i in range(0:n):
            foo( list[i] )

    :param skip: number of elements to skip in the sequence
    :param n_element: number of element to select in the sequence, select all (from skip to end) if n_element < 0
    :param tot_element: total elements in the sequence, normally the length of a list.
    :return: (first_element, last_element, n_elements)
             where last_element is the index of the last element selected + 1

    """
    if skip > tot_element:
        skip = tot_element
    if skip < 0:
        skip=0

    if n_element < 0 or n_element > tot_element - skip:
        n_element = tot_element - skip

    return skip, skip+n_element, n_element



def index_vector(size, one_position):
    import numpy
    v = numpy.zeros(size)
    v[one_position] = 1
    return v