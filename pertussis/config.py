from itertools import product
import numpy as np
import logging
import sys
# Pandas print options
np.set_printoptions(edgeitems=5, linewidth=170, suppress=True, precision=8)

# Logger class
logger = logging.getLogger('pertussis')
logger.setLevel(logging.DEBUG)
formatter = logging.Formatter('%(levelname)-8s %(message)s\n')

sh = logging.StreamHandler()
sh.setFormatter(formatter)
sh.setLevel(logging.INFO)

fh = logging.FileHandler('./pertussis.log', mode='w')
fh.setFormatter(formatter)
fh.setLevel(logging.DEBUG)

logger.addHandler(sh)
logger.addHandler(fh)

# logger.propagate = False
# dean_logger.info("DEAN")
# print (logger.handlers)

# Scenario
_alpha_ap = [1, "like epsilon"]
_omega_ap = [4, 30]
_p = [1 / 250, 0.015]
options = product(range(len(_alpha_ap)), range(len(_omega_ap)), range(len(_p)))
# print (sorted([i for i in options]))
#
SCENARIO = {'main': {'alpha_ap': "like epsilon div 2",
                     'omega_ap': 1 / 30,
                     'p': 1 / 100}}

for i, option in enumerate(sorted(options)):
    # print (i, option)
    SCENARIO[i] = {'alpha_ap': _alpha_ap[option[0]],
                   'omega_ap': 1 / _omega_ap[option[1]],
                   'p': _p[option[2]]}
if __name__ == '__main__':
    from pprint import pprint

    pprint(SCENARIO)
