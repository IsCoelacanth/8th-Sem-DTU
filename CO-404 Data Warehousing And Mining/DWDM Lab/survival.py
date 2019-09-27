from lifelines import KaplanMeierFitter
import matplotlib.pyplot as plt
durations = [11, 74, 71, 76, 28, 92, 89, 48, 90, 39, 63, 36, 54, 64, 34, 73, 94, 37, 56, 76]
event_observed = [True, True, False, True, True, True, True, False, False, True, True,
                  True, True, True, True, True, False, True, False, True]

kmf = KaplanMeierFitter()
kmf.fit(durations, event_observed)
kmf.plot()
plt.show()