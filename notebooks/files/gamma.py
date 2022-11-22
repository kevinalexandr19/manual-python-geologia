####
#### GAMMA
#### Módulo de geoestadística para el modelamiento del variograma
#### Nota: por ahora solo se pueden modelar variogramas 1D

#### Copyright 2022, kevinalexandr19
#### This software may be modified and distributed under the terms of the MIT license.
#### See the LICENSE file for details.

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
import ipywidgets as widgets
from numba import jit
from sklearn.cluster import DBSCAN
from tqdm import tqdm


@jit
# Cálculo de distancia y diferencia cuadrática entre cada par de puntos
def gam1d(values, coords):
    lags = np.array([np.float64(x) for x in range(0)])
    sqvalues = np.array([np.float64(x) for x in range(0)])

    for i in range(1, len(values) + 1):
        dist = np.abs(coords[i:] - coords[:-i])
        sqdiff = np.power(values[i:] - values[:-i], 2)

        lags = np.append(lags, dist)
        sqvalues = np.append(sqvalues, sqdiff)

    return lags, sqvalues

@jit
def linear(h, c0, c, a):
    return c0 + c * (h/a)

@jit
def exponential(h, c0, c, a):
    return c0 + c * (1 - np.exp(-h/a))

@jit
def spherical(h, c0, c, a):
    return c0 + c * ((1.5*(h/a) - 0.5*((h/a)**3)) * (h <= a) + 1 * (h > a))

@jit
def gaussian(h, c0, c, a):
    return c0 + c * (1 - np.exp(-(h**2)/(a**2)))


class Variogram:    
    # Por ahora solo opera con coordenadas en 1D
    def __init__(self, values: np.array, coords: np.array):
        assert len(values) == len(coords), "Los datos de entrada deben tener el mismo tamaño."
        self.values = values
        self.coords = coords
        self.n_samples = len(values)
        
        # Procesamiento de distancias y diferencias cuadráticas
        lags, sqvalues = gam1d(values, coords)
        array = np.array([lags, sqvalues])
        sort = array[:, array[0, :].argsort()]
        self.lags, self.sqvalues = sort[0], sort[1]
        
        self.var = np.var(values)
        self.bins = None
        self.pairs = None
        self.h = None
        self.gamma = None
        self.model = None
        
        
    def binning(self):
        # Agrupa las distancias y diferencias cuadráticas en intervalos (bins)
        lags, sqvalues = self.lags, self.sqvalues
        diff = (lags[1:] - lags[:-1]).mean()
        clustering_lags = []
        for lag in lags:
            clustering_lags.append(lag)
            X = np.array(clustering_lags).reshape(-1, 1)
            clustering = DBSCAN(eps=diff, min_samples=5).fit(X)
            _0 = np.nonzero(clustering.labels_ == 0)
            if clustering.labels_.max() == 4:
                break
        h0 = lags[_0].mean()

        bins = np.arange(1.5 * h0, lags.max() + 1.5 * h0, h0)
        h = bins - 0.5 * h0
        self.bins = bins
        self.h = h

        indices = []
        for b in bins:
            index = (lags <= b).sum()
            indices.append(index)
        split = np.split(np.array([lags, sqvalues]), indices, axis=1)[:-1]

        pairs = []
        gamma = []
        for array in split:
            if array.shape[1] == 0:
                continue
            pairs.append(len(array[1, :]))
            gamma.append(array[1, :].mean() / 2)
        self.pairs = pairs
        self.gamma = gamma
        
        def bins_selector(h0=h0):
            # Selección del intervalo de separación adecuado para los datos
            bins = np.arange(0.5 * h0, self.lags.max() + 0.5 * h0, h0)
            h = bins + 0.5 * h0

            fig, ax = plt.subplots(figsize=(15, 4))
            ax.scatter(h0, y=1, c="red", s=40)
            ax.scatter(x=self.lags, y=[1]*len(self.lags), s=5, c="black")
            for b in bins:
                ax.axvline(x=b, lw=0.7)
            ax.set_xlim(0, (self.lags.max())/2)
            ax.axhline(y=1, lw=0.5)
            ax.set_yticks([])
            ax.set_title("Intervalos de separación (lag classes)", fontsize=16)
            ax.set_xlabel("Distancia entre puntos", fontsize=16)
            ax.annotate(xy=(h0, 1), xytext=(h0, 1.02), text="$h_{0}$", ha="center",
                        fontsize=10, arrowprops=dict(arrowstyle="-|>", color="black"))
            plt.show()

        widgets.interact(bins_selector, h0=(lags.min(), lags.max()/2))
        
        
    def fitting(self):
        # Ajuste del variograma
        h, gamma, var = self.h, self.gamma, self.var
        hmax = self.h.max()        
        
        data = []
        for func, name in zip([linear, exponential, spherical, gaussian], ["linear", "exponential", "spherical", "gaussian"]):
            print(f"Procesando información del modelo: {name}")
            for n in tqdm(range(5, self.n_samples)):
                gamma_min = min(gamma[:n])
                h_min = min(h[:n])
                
                # Modelo
                popt, pcov = curve_fit(func, h[:n], gamma[:n], p0=[np.mean(gamma), np.mean(gamma), np.mean(h)],
                                       bounds=(min([gamma_min, h_min]), np.inf), maxfev=10000) 
                # R2
                residuals = gamma[:n] - func(h[:n], *popt)
                ss_res = np.sum(residuals ** 2)
                ss_tot = np.sum((gamma[:n] - np.mean(gamma[:n])) ** 2)
                r2 = 1 - (ss_res / ss_tot)
                
                # Generación de datos para el modelamiento
                if func == linear:
                    c0 = popt[0]
                    c = np.nan
                    a = h[n-1]
                    data.append([name, n, c0, c, a, r2, ss_res, popt])
                elif func == exponential:
                    c0 = popt[0]
                    c = popt[1]
                    a = 3 * popt[2]
                    data.append([name, n, c0, c, a, r2, ss_res, popt])
                elif func == spherical:
                    c0 = popt[0]
                    c = popt[1]
                    a = popt[2]
                    data.append([name, n, c0, c, a, r2, ss_res, popt])
                elif func == gaussian:
                    c0 = popt[0]
                    c = popt[1]
                    a = (3 ** 0.5) * popt[2]
                    data.append([name, n, c0, c, a, r2, ss_res, popt])
        
        data = pd.DataFrame(data, columns=["model", "n_points", "c0", "c", "a", "r2", "rss", "popt"])
        r2max = data[(data["r2"] == data["r2"].max())].values[0]
        
        self.data = data  
        self.model = r2max[0]
        self.model_params = r2max[1:-2]
             
        def model_fitter(model=self.model, n_points=self.model_params[0]):
            # Selección del modelo a utilizar en el ajuste del variograma
            data = self.data
            params = data[(data["model"] == model) & (data["n_points"] == n_points)][["c0", "c", "a", "r2", "rss", "popt"]].values[0]
            c0, c, a, r2, rss, popt = params
            
            if model == "linear":
                modelfunc = linear
            elif model == "exponential":
                modelfunc = exponential
            elif model == "spherical":
                modelfunc = spherical
            elif model == "gaussian":
                modelfunc = gaussian
            
            print(f"R2: {r2:.3f} RSS: {rss:.3f}")
            print(f"Puntos de ajuste: {n_points}")
            print(f"varianza: {var:.3f}")
            print(f"C0: {c0:.3f}")
            if model != "linear":
                print(f"C: {c:.3f}")
                print(f"sill: {c0 + c:.3f}")
            print(f"alcance: {a:.3f}")
            
            fig, ax = plt.subplots(figsize=(10, 5))
            ax.scatter(h[:n_points], gamma[:n_points], c="white", s=10, marker="s", edgecolor="blue")
            ax.plot(h[:n_points], modelfunc(h[:n_points], *popt), c="black", lw=1.2)
            ax.axhline(y=var, linewidth=0.8, c="red")
            ax.axvline(x=a, linewidth=0.8, c="red")
            ax.set_xlim(left=0, right=hmax)
            ax.set_ylim(bottom=0, top=2*var)
            ax.grid(lw=0.5)
            plt.show()
        
        widgets.interact(model_fitter, model=["linear", "exponential", "spherical", "gaussian"], n_points=(5, self.n_samples-1))
        