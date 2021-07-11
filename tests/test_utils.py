from naprima.utils import *
import numpy as np

def test_listify_return_type():
    l = [1,2,3]
    assert isinstance(listify(l),list)
    assert isinstance(listify(np.array(l)),list)
    assert isinstance(listify(np.array([l])),list)
    assert isinstance(listify(np.array(l).reshape((1,1,1,1,1,3))),list)
    assert listify(l) == l
    assert listify(np.array(l)) == l
    assert listify(np.array([l])) == l
    assert listify(np.array(l).reshape((1,1,1,1,1,3))) == l
