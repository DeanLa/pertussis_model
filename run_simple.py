from scipy.integrate import odeint
import numpy as np
import matplotlib.pyplot as plt

from pertussis.simple import main_model
from pertussis.params.simple import collect_state0
from pertussis.charts import draw_model

plt.style.use('ggplot')

# Compartments
state_0 = collect_state0()

# state_0 = (S,Ia,Is,V,R)
print(state_0)
#
t_start = 1948
step = 1
t_end = t_start + (2025 - t_start) * step
t_range = np.arange(t_start, t_end + 0.1, 0.1)
RES = odeint(main_model, state_0, t_range, args=(step,))
#
x = (t_range - t_start) / step + t_start
# draw_model(x, RES, ["S", "V", "Is", "Ia", "R"])
draw_model(x, RES, ["S", "I", "R"], split=1)
plt.tight_layout()
plt.show()
