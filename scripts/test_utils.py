import numpy as np

def hvstack(one, two, three, four):
    final = np.vstack(
         (
             np.hstack((one.squeeze(), two.squeeze())),
             np.hstack((three.squeeze(), four.squeeze()))
         )
     )
    return final

from itertools import zip_longest
def grouper(n, iterable, fillvalue=None):
    "grouper(3, 'ABCDEFG', 'x') --> ABC DEF Gxx"
    args = [iter(iterable)] * n
    return zip_longest(fillvalue=fillvalue, *args)
