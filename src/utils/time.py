import numpy as np
from datetime import datetime
from datetime import timedelta

class AverageTimer:
    def __init__(self):
        self.timings = []

    def start(self):
        self.t0 = datetime.now()

    def end(self, engine=None):
        self.t = datetime.now() - self.t0
        self.timings.append(self.t)
        if engine is not None and engine.state.iteration % 10 == 0:
            print(
                f"Iteration {engine.state.iteration}, time per batch: {self.t}, average time per batch in milliseconds: {np.mean(np.array(self.timings)/timedelta(milliseconds=1))}"
            )
    def print_avg(self):
        print(f"Average time per batch in milliseconds: {np.mean(np.array([t/timedelta(milliseconds=1) for t in self.timings]))}")

class TrainingTimer:
    def __init__(self):
        self.timings = []

    def start(self):
        self.t0 = datetime.now()

    def end(self):
        self.t = datetime.now() - self.t0
        self.timings.append(self.t)

    def print_avg(self):
        print(f"Average time per epoch in seconds: {np.mean(np.array([t/timedelta(seconds=1) for t in self.timings]))}")
        