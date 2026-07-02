import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import linregress

class VoceCalibrator:

    def __init__(
            self,
            test
    ):
        
        self.test = test

    def find_modulus_region(
            self,
            stress,
            strain,
            window=40,
    ):
        
        idx = np.where(strain > 0.01)[0]

        if len(idx) == 0:
            raise RuntimeError
        else:
            idx = idx[0]

        best_r2 = -np.inf
        best = None

        for i in range(len(stress) - window):
            for j in range(i + window, len(stress)):

                x = strain[i:j]
                y = stress[i:j]
                slope, intercept, rvalue, _, _ = linregress(x, y)

                r2 = rvalue**2
                if slope <= 0:
                    continue

                if r2 > best_r2:
                    best_r2 = r2
                    best = (slope, intercept, i, j)
        
        if best is None:
            raise RuntimeError(
                f"Coult not identify linear region. Best R*R={best_r2:.4f}"
            )
        
        E, intercept, start, stop = best
        print("Results from linear region identification:")
        print(f"    {best_r2 = :.4f}    {E = :.1f} GPa")

    def correct_initia_stiffness(self):

        E_m, b, start, stop = self.find_modulus_region()