from itertools import product
import numpy as np
import logging
import sys
from datetime import datetime
# Pandas print options
np.set_printoptions(precision=4, suppress=True, linewidth=80)

file_name = str(datetime.now()).replace(" ","-").replace(":","-")
file_name = file_name[:file_name.find('.')]+'.log'
file_name = 'my_log.log'
# print (file_name)
# Logger class
logger = logging.getLogger('pertussis')
logger.setLevel(logging.DEBUG)
formatter = logging.Formatter('%(levelname)-s, %(message)s')

sh = logging.StreamHandler()
sh.setFormatter(formatter)
sh.setLevel(logging.CRITICAL)

fh = logging.FileHandler('./log/'+file_name, mode='a')
fh.setFormatter(formatter)
fh.setLevel(logging.DEBUG)


# progress = logging.getLogger('progress')
# pb = logging.FileHandler('./progress.log', mode='w')
# pb.setFormatter(formatter)
# pb.setLevel(logging.DEBUG)

logger.addHandler(sh)
# logger.addHandler(fh)
# progress.addHandler(pb)

# logger.propagate = False
# dean_logger.info("DEAN")
# print (logger.handlers)

# Scenario
_alpha_ap = [1, "like1"]
_omega_ap = [4, 30]
_p = [1 / 250, 0.015]
options = product(range(len(_alpha_ap)), range(len(_omega_ap)), range(len(_p)))
# print (sorted([i for i in options]))
#
SCENARIO = {'main': {'alpha_ap': "like1.5",
                     'omega_ap': 1 / 18, # Years
                     'p': 1 / 100}}

for i, option in enumerate(sorted(options)):
    # print (i, option)
    SCENARIO[i] = {'alpha_ap': _alpha_ap[option[0]],
                   'omega_ap': 1 / _omega_ap[option[1]],
                   'p': _p[option[2]]}
if __name__ == '__main__':
    from pprint import pprint

    pprint(SCENARIO)
