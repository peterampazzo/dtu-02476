import pickle
import time

import numpy as np
import torch

import multiprocessing

nb_core = multiprocessing.cpu_count()

core = [i for i in range(0, nb_core + 1)]
deviations = []
timings = []

if __name__ == "__main__":
    for i in core:
        res = []
        for j in range(0, 5):
            start = time.time()

            train = torch.load("data/processed/train.pt")
            train_set = torch.utils.data.DataLoader(
                train, batch_size=90, shuffle=True, num_workers=i
            )

            for batch_idx, batch in enumerate(train_set):
                if batch_idx > 10:
                    break

            end = time.time()

            res += [end - start]

            res = np.array(res)
        print("Timing:", np.mean(res), "+-", np.std(res))
        timings += [np.mean(res)]
        deviations += [np.std(res)]

    with open("reports/distributed_datas/timings", "wb") as fp:  # Pickling
        pickle.dump(timings, fp)
    with open("reports/distributed_datas/deviations", "wb") as fp:  # Pickling
        pickle.dump(deviations, fp)
