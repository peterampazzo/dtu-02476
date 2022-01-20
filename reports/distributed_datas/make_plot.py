import matplotlib.pyplot as plt
import pickle

import multiprocessing

nb_core = multiprocessing.cpu_count()

core = [i for i in range(0,nb_core+1)]

with open("reports/distributed_datas/timings", "rb") as fp:   
    timings = pickle.load(fp)

with open("reports/distributed_datas/deviations", "rb") as fp:   
    deviations = pickle.load(fp)

print(deviations)

plt.bar(core, timings, yerr=deviations, align='center', alpha=0.5, ecolor='black', capsize=10)
plt.ylabel('Computing time of loading batchs')
plt.xticks(core,core)
plt.title('Number of cores')

# Save the figure and show
plt.tight_layout()
plt.savefig('reports/distributed_datas/Distributed_data_computing_time.png')
