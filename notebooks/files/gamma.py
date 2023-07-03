####
#### GAMMA v1.2 Junio 2023
#### Módulo de geoestadística para el modelamiento del variograma
#### Nota: solo variograma 1D, el variograma 2D está en desarrollo

#### Copyright 2023, kevinalexandr19
#### This software may be modified and distributed under the terms of the MIT license.
#### See the LICENSE file for details.

import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
import ipywidgets as widgets
from numba import jit
from sklearn.cluster import DBSCAN
from IPython.display import Markdown


###############################################################################
# Cálculo de distancia y diferencia cuadrática entre cada par de puntos para 1D
@jit
def gam1d(values, coords):
    lags = np.array([np.float64(x) for x in range(0)])
    sqvalues = np.array([np.float64(x) for x in range(0)])

    for i in range(1, len(values) + 1):
        dist = np.abs(coords[i:] - coords[:-i])
        sqdiff = np.power(values[i:] - values[:-i], 2)

        lags = np.append(lags, dist)
        sqvalues = np.append(sqvalues, sqdiff)

    return lags, sqvalues


###############################################################################
# Cálculo de distancia y diferencia cuadrática entre cada par de puntos para 2D
@jit
def gam2d(values, coords):
    lags = np.array([np.float64(x) for x in range(0)])
    sqvalues = np.array([np.float64(x) for x in range(0)])
    angles = np.array([np.float64(x) for x in range(0)])

    for i in range(1, len(values) + 1):
        dist = coords[:, i:] - coords[:, :-i]
        az = np.degrees(np.arctan2(dist[0, :], dist[1, :]))
        dist = np.power(np.power(dist[0, :], 2) + np.power(dist[1, :], 2), 0.5)
        sqdiff = np.power(values[i:] - values[:-i], 2)

        lags = np.append(lags, dist)
        sqvalues = np.append(sqvalues, sqdiff)
        angles = np.append(angles, az)
    
    return lags, sqvalues, angles


###################################################
# Modelos de ajuste para el variograma experimental
@jit
def linear(h, c0, c, a):
    return c0 + c * (h/a)

@jit
def exponential(h, c0, c, a):
    return c0 + c * (1 - np.exp(-h/a))

@jit
def spherical(h, c0, c, a):
    return c0 + c * (((1.5 * (h / a)) - (0.5 * ((h / a) ** 3))) * (h <= a) + 1 * (h > a))

@jit
def gaussian(h, c0, c, a):
    return c0 + c * (1 - np.exp(-(h ** 2) / (a ** 2)))


###################################################
# Métricas de evaluación de ajuste
@jit
def r2_score(gamma: np.array, prediction: np.array):
    ss_res = np.sum((gamma - prediction) ** 2)
    ss_tot = np.sum((gamma - np.mean(gamma)) ** 2)
    r2 = 1 - (ss_res / ss_tot)
    return r2

@jit
def mse_score(gamma: np.array, prediction: np.array):
    mse = np.mean((gamma - prediction) ** 2)
    return mse


###################################################
# Algoritmo de búsqueda de lag clusters
def search_clusters(lags, diff, n_steps):
    if len(lags) > 1000:
        for i in range(2, n_steps+1):
            X = lags[:1000*i].reshape(-1, 1)
            clustering = DBSCAN(eps=diff, min_samples=5).fit(X)
            if clustering.labels_.max() >= 3:
                _0 = np.nonzero(clustering.labels_ == 0)
                lagDist = lags[_0].mean()
                return lagDist
    else:
        X = lags.reshape(-1, 1)
        clustering = DBSCAN(eps=diff, min_samples=5).fit(X)
        _0 = np.nonzero(clustering.labels_ == 0)
        lagDist = lags[_0].mean()
        return lagDist


#############################################################################################
# Widget que selecciona la distancia de separación entre puntos en un variograma experimental
class LagDistSelector(object):
    def __init__(self, variogram):
        self.Variogram = variogram
        self.lags = variogram.lags
        self.lagDist = variogram.lagDist
        
##### Gráfico que muestra las distancias de separación para el variograma experimental 
    def lagDistPlot(self, lagDist, distance):
        lags = self.lags
        
        # Selección del intervalo de separación adecuado para los datos
        bins = np.arange(0.5 * lagDist, lags.max() + 0.5 * lagDist, lagDist)
        h = bins + 0.5 * lagDist

        # Figura
        fig, ax = plt.subplots(figsize=(15, 4))

        # Dispersión de puntos en una recta numérica
        ax.scatter(lagDist, y=1, c="red", s=120)
        ax.scatter(x=lags, y=[1]*len(lags), s=40, c="black", alpha=0.8)

        # Intervalos de separación en lineas verticales azules
        for b in bins:
            ax.axvline(x=b, lw=0.7)

        # Límites de la figura
        ax.set_xlim(0, distance)

        # Detalles adicionales
        ax.set_yticks([])
        ax.axhline(y=1, lw=0.5)

        # Texto
        ax.set_title("Intervalos de separación (lag classes)", fontsize=16)
        ax.set_xlabel("Distancia entre puntos", fontsize=16)
        ax.annotate(xy=(lagDist, 1), xytext=(lagDist, 1.02), text="$h_{0}$", ha="center",
                    fontsize=15, arrowprops=dict(arrowstyle="-|>", color="black"))
        plt.show()

##### Método para desplegar el widget de selección    
    def lagDistWidget(self):
        title = """# <font size="5">Ajuste de distancia de separación </font>"""
                
        # Resumen texto
        line1 = f"Total de pares: {len(self.lags)}"
        line2 = f"Distancia mínima: {self.lags.min():.2f}"
        line3 = f"Distancia máxima: {self.lags.max():.2f}"
        lines = widgets.VBox([widgets.HTML(line1), widgets.HTML(line2), widgets.HTML(line3)])
        lines.layout.padding = "10px 0 0 0"
        
        # Widget para seleccionar el lagDist
        lagWidget = widgets.FloatSlider(value=self.Variogram.lagDist, min=self.lags.min(), max=self.lags.max()/2,
                                        description="lagDist")
        lagWidget.layout.padding = "30px 0px 0px 0px"
        
        # Botón para asignar el lagDist seleccionado
        button = widgets.Button(description="Asignar lagDist", button_style="", 
                                icon="save", style={"button_color": "green"})
        button.layout.padding = "0.5% 5% 5% 5%"
        button.layout.width = "80%"

        def button_click(b):
            with output:
                self.lagDist = lagWidget.value
                self.Variogram.updateLagDist()

        button.on_click(button_click)
        
        # Widget para seleccionar la distancia máxima en la figura
        distanceWidget = widgets.FloatSlider(value=5*self.Variogram.lagDist, min=2*self.lags.min(), max=self.lags.max()/2,
                                             description="Distancia")
        distanceWidget.layout.padding = "30px 0px 0px 0px"
        
        # Output de la figura
        output = widgets.interactive_output(self.lagDistPlot, {"lagDist": lagWidget, "distance": distanceWidget})
        output.layout.width = "99%"
        output.layout.padding = "0 0 0 0%"
        output.clear_output(wait = True)
        
        # Caja izquierda
        leftBox = widgets.VBox([button, lines])
        leftBox.layout.width = "50%"
        leftBox.layout.padding = "10px 0 30px 30px"
        
        # Caja derecha
        rightText = widgets.HTML("""<font size="5">Parámetros</font>""")
        rightText.layout.padding = "0 0 0 20px"
        rightBox = widgets.VBox([rightText, lagWidget, distanceWidget])
        rightBox.layout.width = "50%"
        rightBox.layout.padding = "10px 0 30px 0px"
        
        # Caja izquierda y derecha
        widgetBox = widgets.HBox([leftBox, rightBox])
        widgetBox.layout.width = "80%"
        widgetBox.layout.align_items = "center"
        
        # Caja principal
        container = widgets.VBox([output, widgetBox])        
        container.layout.padding = "0% 0 0% 1.5%"
        container.layout.width = "900px"
        container.layout.border = "solid 5px green"
        
        # Desplegar los widgets
        display(Markdown(title), container)    

        
#########################
# Variograma experimental
class Variogram(object):    
    # Por ahora solo opera con coordenadas en 1D
    """Variograma experimental para análisis geoestadístico en 1D, admite un mínimo de 10 muestras.
    
       Contiene funcionalidades para ajustar los intervalos de separación (lag classes)
       y el modelo del variograma.
       Los intervalos de separación son equidistantes.
       Los modelos pueden ser de tipo lineal, exponencial, esférico o gausiano.
        
       Parameters
       -----------
       values: ndarray, representa los valores de la variable, siempre tiene la forma (n,)
               donde n es el número de muestras.
       coords: ndarray, representa las coordenadas de la variable, dependiendo del número 
               de dimensiones, puede tener la forma (n,), (n, 2) o (2, n) donde n es el 
               número de muestras.
       lag_dist: distancia de separación usada para calcular los puntos del variograma experimental.
       
    """
    
    def __init__(self, values: np.ndarray, coords: np.ndarray, lagDist=None):
        
######### Verifica que los valores de la variable sean de la forma (n,)
        assert values.ndim == 1, "Los valores de la variable no están en una sola dimensión."
        self.n = len(values) # Número de muestras
        assert self.n >= 10, "Cantidad insuficiente de muestras para modelar el variograma experimental."
        
######### Verifica que la distancia de separación sea un número
        if lagDist:
            assert type(lagDist) in [int, float], "La distancia de separación debe ser un número (int o float)."
        
######### Verifica que los valores y las coordenadas contengan el mismo número de muestras
######### También asigna la dimensión del variograma experimental (1D, 2D, 3D)
        if coords.ndim == 1:
            assert len(coords) == self.n, "Los valores y coordenadas de la variable no tienen el mismo número de filas."
            self.dim = "1D"
        else:
            shape = coords.shape
            assert len(shape) == 2, "Las coordenadas deben tener la forma (n, 2), (2, n), (n, 3) o (3, n) donde n es el número de muestras."
            argdim, argval = np.argmin(shape), np.argmax(shape)
            assert len(values) == shape[argval], "Valores y coordenadas de la variable no tienen el mismo número de muestras."
            self.dim = f"{shape[argdim]}D"
            
            # Si tiene la forma (n, 2) o (n, 3), cambiar a (2, n) o (3, n)
            if argdim == 1:
                coords = coords.T
        
######### Valores y coordenadas de la variable
        self.values = values
        self.coords = coords
        
######### Otras variables
        self.var = np.var(values) # Varianza
        self.model = None         # Modelo     
        
######### Procesamiento de variograma 1D
        if self.dim == "1D":
            # Procesamiento de distancias y diferencias cuadráticas
            lags, sqvalues = gam1d(values, coords)
            array = np.array([lags, sqvalues])
            sort = array[:, array[0, :].argsort()] # distancias ordenadas de menor a mayor
            self.lags, self.sqvalues = sort[0], sort[1]
            
            # Distancia mínima para establecer lag classes
            if not lagDist:
                n_steps = len(lags) // 1000
                diff = (lags[1:] - lags[:-1]).mean()
                lagDist = search_clusters(lags, diff, n_steps)
                self.lagDist = lagDist
                        
            # Agrupa las distancias y diferencias cuadráticas en intervalos (bins)
            bins = np.arange(1.5 * lagDist, lags.max() + 1.5 * lagDist, lagDist)
            h = bins - 0.5 * lagDist
            indices = []
            for b in bins:
                index = (lags <= b).sum()
                indices.append(index)
            split = np.split(np.array([lags, sqvalues]), indices, axis=1)[:-1]
            self.bins = bins
            self.h = h

            # Extrae los valores de gamma y pares para el variograma experimental
            pairs, gamma = [], []
            for array in split:
                if len(array[1, :]) == 0:
                    continue
                pairs.append(len(array[1, :]))
                if len(array[1, :]) == 1:
                    gamma.append(float(array[1, :]))
                else:
                    gamma.append(array[1, :].mean() / 2)
            self.pairs = np.array(pairs)
            self.gamma = np.array(gamma)
            
            # Selector de distancias
            self.lagDistSelector = LagDistSelector(self)
            
            # Unidades de escala para la semivarianza y la distancia
            self.var_unit = int(np.floor(np.log10(self.var)))
            self.dist_unit = int(np.floor(np.log10(self.h.max())))
            self.var = np.round(self.var, 2 - self.var_unit)
                
######### Procesamiento de variograma 2D
        elif self.dim == "2D":
            # Procesamiento de distancias y diferencias cuadráticas
            lags, sqvalues, angles = gam2d(values, coords)
            array = np.array([lags, sqvalues, angles])
            sort = array[:, array[0, :].argsort()] # distancias ordenadas de menor a mayor
            self.lags, self.sqvalues, self.angles = sort[0], sort[1], sort[2]
        
######### Procesamiento de variograma 3D
        elif self.dim == "3D":
            pass

        
##### Descripción del objeto creaodo
    def __str__(self):
        str1 = f""">>> Variograma experimental {self.dim}
--------------------------------
Total de muestras (n): {self.n}
Total de pares (pairs): {len(self.lags)}
Intervalos (bins): {len(self.bins)}
Distancia (lagDist): {self.lagDist}
--------------------------------
"""
        
        if self.model != None:
            model, n_points = self.model["model"], self.model["n_points"]
            c0, c, a, sill = self.model["c0"], self.model["c"], self.model["a"], self.model["sill"]
            popt = self.model["popt"]
            r2, mse = self.model["r2"], self.model["mse"]

            str2 = f""">>> Ajuste de modelo de variograma
--------------------------------
Modelo (model): {model}
Pepita o Nugget (C0): {c0}
Sill parcial (C): {c}
Sill (C0 + C): {sill}
Alcance (a): {a}
Parámetros de función: {popt}
R2: {r2}   MSE: {mse}"""

            return str1 + str2
        else:
            return str1
        
        
##### Método para actualizar la distancia de separación
    def updateLagDist(self):
        lagDist = self.lagDistSelector.lagDist
        lags, sqvalues = self.lags, self.sqvalues
        
        # Agrupa las distancias y diferencias cuadráticas en intervalos (bins)
        bins = np.arange(1.5 * lagDist, lags.max() + 1.5 * lagDist, lagDist)
        h = bins - 0.5 * lagDist
        indices = []
        for b in bins:
            index = (lags <= b).sum()
            indices.append(index)
        split = np.split(np.array([lags, sqvalues]), indices, axis=1)[:-1]
        self.lagDist = lagDist
        self.bins = bins
        self.h = h

        # Extrae los valores de gamma y pares para el variograma experimental
        pairs, gamma = [], []
        for array in split:
            if len(array[1, :]) == 0:
                continue
            pairs.append(len(array[1, :]))
            if len(array[1, :]) == 1:
                gamma.append(float(array[1, :]))
            else:
                gamma.append(array[1, :].mean() / 2)        
        self.pairs = np.array(pairs)
        self.gamma = np.array(gamma)

##### Método para mostrar el widget que permite elegir la distancia de separación
    def selectLagDist(self):
        self.lagDistSelector.lagDistWidget()      

        
##### Método para ajustar el modelo del variograma experimental        
    def fitting(self):
        # Ajuste del variograma
        h, gamma, var = self.h, self.gamma, self.var
        hmax = h.max()
        pairs = self.pairs
        sigma = 1 / pairs
        
        # Generación de datos para el modelamiento
        table = []
        fit_functions = [linear, exponential, spherical, gaussian]
        fit_names = ["linear", "exponential", "spherical", "gaussian"]
        
        for func, name in zip(fit_functions, fit_names):
            for n in range(5, len(gamma)): # Mínimo 5 puntos de ajuste
                # Modelo
                p0 = [0, var * (h[n] / hmax), h[n]]
                popt, pcov = curve_fit(func, h[:n], gamma[:n], sigma=sigma[:n], p0=p0, bounds=(0, np.inf))      
                # R2 y MSE
                prediction = func(h[:n], *popt)
                r2 = r2_score(gamma[:n], prediction)
                mse = mse_score(gamma[:n], prediction)
                
                # Acotación de parámetros
                c0 = popt[0]
                if func == linear:
                    # El sill y el alcance coinciden con distancia máxima de ajuste
                    c = linear(h[n-1], *popt) - c0
                    a = h[n-1]
                    table.append([name, n, c0, c, a, r2, mse, popt])
                elif func == exponential:
                    c = popt[1]
                    a = popt[2] * 3          # Alcance efectivo h = 3 * a
                    table.append([name, n, c0, c, a, r2, mse, popt])
                elif func == spherical:
                    c = popt[1]
                    a = popt[2]
                    table.append([name, n, c0, c, a, r2, mse, popt])
                elif func == gaussian:
                    c = popt[1]
                    a = popt[2] * (3 ** 0.5) # Alcance efectivo h = sqrt(3) * a
                    table.append([name, n, c0, c, a, r2, mse, popt])
        
        # DataFrame con los datos de ajuste
        data = pd.DataFrame(table, columns=["model", "n_points", "c0", "c", "a", "r2", "mse", "popt"])
        
        # Redondear los datos de c0, c, a, r2, mse
        data["c0"] = data["c0"].apply(lambda x: np.round(x, 2 - self.var_unit))
        data["c"] = data["c"].apply(lambda x: np.round(x, 2 - self.var_unit))
        data["a"] = data["a"].apply(lambda x: np.round(x, 2 - self.dist_unit))
        data["r2"] = data["r2"].apply(lambda x: np.round(x, 3))
        data["mse"] = data["mse"].apply(lambda x: np.round(x, 3 - self.var_unit))
        data["sill"] = data["c0"] + data["c"]
        data["sill"] = data["sill"].apply(lambda x: np.round(x, 2 - self.var_unit))
        
        # Guardar el dataframe
        self.data = data                
        
        # Selección de valores óptimos de ajuste
        self.model = data.sort_values(by=["r2", "n_points"], ascending=False).iloc[0, :].copy().to_dict()
        self.current_model = None
        
######### Función que ajusta un modelo específico al variograma experimental
        default_adjust = [self.model["c0"], self.model["c"], self.model["a"]]
        def model_fitter(model=self.model["model"], n_points=self.model["n_points"], adjust=default_adjust, auto=True):
            # Selección del modelo a utilizar en el ajuste del variograma
            data = self.data
            mask = (data["model"] == model) & (data["n_points"] == n_points)
            cols = ["c0", "c", "a", "r2", "mse", "popt", "sill"]
            c0, c, a, r2, mse, popt, sill = data[mask][cols].copy().values[0]
            popt = popt.copy()
            
            # Selección de modelo
            if model == "linear":
                modelfunc = linear
            elif model == "exponential":
                modelfunc = exponential
            elif model == "spherical":
                modelfunc = spherical
            elif model == "gaussian":
                modelfunc = gaussian
            
            # Ajuste manual
            if not auto:
                popt[0] = adjust[0]
                popt[1] = adjust[1]
                if model == "exponential":
                    popt[2] = adjust[2] / 3
                elif model == "gaussian":
                    popt[2] = adjust[2] / (3 ** 0.5)
                else:
                    popt[2] = adjust[2]
                    
                # c0, c, a
                c0 = popt[0]
                if model == "linear":
                    # El sill y el alcance coinciden con distancia máxima de ajuste
                    c = linear(h[n-1], *popt) - c0
                    a = h[n-1]
                elif model == "exponential":
                    c = popt[1]
                    a = popt[2] * 3          # Alcance efectivo h = 3 * a
                elif model == "spherical":
                    c = popt[1]
                    a = popt[2]
                elif model == "gaussian":
                    c = popt[1]
                    a = popt[2] * (3 ** 0.5) # Alcance efectivo h = sqrt(3) * a
                
                # r2 y mse
                prediction = modelfunc(h[:n_points], *popt)
                r2 = r2_score(gamma[:n_points], prediction)
                mse = mse_score(gamma[:n_points], prediction)
            
            self.current_model = {"model": model, "n_points": n_points,
                                  "c0": c0, "c": c, "a": a,
                                  "r2": r2, "mse": mse, 
                                  "popt": popt, "sill": sill}            
            
            # Figura principal
            fig, ax = plt.subplots(figsize=(10, 5))
            
            # Semivariograma experimental
            ax.scatter(h[:n_points], gamma[:n_points], c="white", s=10, marker="s", edgecolor="blue")
            
            # Modelo de semivariograma
            ax.plot(h[:n_points], modelfunc(h[:n_points], *popt), c="black", lw=1.2)
            
            # Etiquetas del eje x e y
            ax.set_xlabel("Distancia de separación (h)", fontsize=18)
            ax.set_ylabel("Semivariograma", fontsize=18)
            
            ax.axhline(y=var, linewidth=0.8, c="red") # Varianza
            ax.axvline(x=a, linewidth=0.8, c="red")   # Alcance
            
            # Límites del gráfico
            ax.set_xlim(left=0, right=hmax)
            ax.set_ylim(bottom=0, top=2*var)
            
            ax.grid(lw=0.5) # Grilla
            
            plt.show()
            
            print(f"Modelo: {model}     Puntos de ajuste: {n_points}")
            print(f"R2: {r2:.3f}    MSE: {mse:.3f}    Varianza: {var}")
            
######### Widget de ajuste del modelo
        
        # Escala de semivarianza y distancia
        step_var = 10 ** (self.var_unit - 2)
        step_dist = 10 ** (self.dist_unit - 2)
    
        # Título del widget
        title = widgets.HTML(value="<h2>Gamma - Ajuste de Variograma</h2>",
                             layout=dict(width="80%", margin="0 0 0 10%")
                             )
        
        # Output inicial
        output = widgets.Output(layout=dict(margin="3% 2% 3% 2%",
                                            justify_content="center", align_items="center", display="flex"))
        with output:
            model_fitter()
        
        # Widget de model
        _model = widgets.Dropdown(value=self.model["model"], options=["linear", "exponential", "spherical", "gaussian"],
                                  description="Modelo:", 
                                  layout=dict(width="80%", margin="3% 0 0 10%")
                                  )
        
        # Widget de n_points
        _n_points = widgets.IntSlider(value=self.model["n_points"], min=5, max=len(gamma)-1, 
                                      step=1, description="Puntos:",
                                      layout=dict(width="80%", margin="2% 0 4% 10%")
                                      )
        
        # Widget de autoajuste
        auto = widgets.Checkbox(value=True, description="Autoajuste", 
                                layout=dict(width="80%", margin="0 0 0 10%")
                                )
        
        # Widget de c0
        _nugget = widgets.BoundedFloatText(value=self.model["c0"], min=0.0, max=sys.float_info.max, disabled=True,
                                           step=step_var, description="Pepita (C0):",
                                           layout=dict(width="80%", margin="0 0 0 10%")
                                           )
        
        # Widget de c
        _c = widgets.BoundedFloatText(value=self.model["c"], min=0, max=sys.float_info.max, disabled=True,
                                      step=step_var, description="Sill Parcial (C):",
                                      layout=dict(width="80%", margin="0 0 0 10%")
                                     )
        
        # Widget de sill
        _sill = widgets.BoundedFloatText(value=self.model["sill"], min=0, max=sys.float_info.max, disabled=True,
                                         step=step_var, description="Sill (C + C0):",
                                         layout=dict(width="80%", margin="0 0 0 10%")
                                         )
        
        # Widget de alcance
        _range = widgets.BoundedFloatText(value=self.model["a"], min=self.h[0], max=sys.float_info.max, disabled=True,
                                          step=step_dist, description="Alcance (a)",
                                          layout=dict(width="80%", margin="0 0 0 10%")
                                          )
        
        # Interactividad de model y n_points
        def on_modelpoints_change(change):
            with output:
                if auto.value: # Autoajuste activado
                    model, n_points = _model.value, _n_points.value
                    data = self.data
                    mask = (data["model"] == model) & (data["n_points"] == n_points)
                    cols = ["c0", "c", "sill", "a"]
                    c0, c, sill, a = data[mask][cols].copy().values[0]
                    
                    _nugget.value = c0
                    _c.value = c
                    _sill.value = sill
                    _range.value = a
                    adjust = [_nugget.value, _c.value, _range.value]
                else:
                    adjust = [_nugget.value, _c.value, _range.value]
                    _sill.value = np.round(np.add(_nugget.value, _c.value), 2 - self.var_unit)
                    
                output.clear_output() # Limpiar el gráfico anterior
                model_fitter(model=_model.value, n_points=_n_points.value, adjust=adjust, auto=auto.value)
        
        
        _model.observe(on_modelpoints_change, names="value")
        _n_points.observe(on_modelpoints_change, names="value")
        
        # Interactividad de c0, c y a
        def on_value_change(change):
            if auto.value:
                pass
            else:
                with output:
                    adjust = [_nugget.value, _c.value, _range.value]
                    _sill.value = np.round(np.add(_nugget.value, _c.value), 2 - self.var_unit)

                    output.clear_output()  # Limpiar el gráfico anterior
                    model_fitter(model=_model.value, n_points=_n_points.value, adjust=adjust, auto=auto.value)
                
        
        _nugget.observe(on_value_change, names="value")
        _c.observe(on_value_change, names="value")
        _range.observe(on_value_change, names="value")
        
        # Interactividad de autoajuste
        def auto_activation(change):
            with output:
                model = _model.value
                n_points = _n_points.value
                mask = (data["model"] == model) & (data["n_points"] == n_points)
                cols = ["c0", "c", "a", "r2", "mse", "popt", "sill"]
                c0, c, a, r2, rss, popt, sill = data[mask][cols].copy().values[0]
                popt = popt.copy()
                
                if change.new:
                    _nugget.value, _c.value, _sill.value, _range.value = c0, c, sill, a
                    output.clear_output()  # Limpiar el gráfico anterior
                    model_fitter(model=model, n_points=n_points, adjust=[c0, c, a], auto=True)
                    _nugget.disabled = True
                    _c.disabled = True
                    _range.disabled = True
                    
                else:
                    _nugget.value, _c.value, _sill.value, _range.value = c0, c, sill, a
                    output.clear_output()  # Limpiar el gráfico anterior
                    model_fitter(model=model, n_points=n_points, adjust=[c0, c, a], auto=False)
                    _nugget.disabled = False
                    _c.disabled = False
                    _range.disabled = False
        
        auto.observe(auto_activation, names="value")
        
        # Botón para guardar los ajustes
        save_button = widgets.Button(description="Guardar parámetros", icon="save",
                                     style=dict(button_color="green", font_weight="bold"),
                                     layout=dict(width="60%", margin="25px 0 0 20%")
                                     )
        
        # Interactividad de botón de guardar
        def on_save_click(b):
            self.model = self.current_model
        
        save_button.on_click(on_save_click)

        # Estructura del widget izquierdo
        left = widgets.VBox([output], layout=dict(width="60%", border="2px solid lightgreen")) 
        
        # Estructura del widget derecho
        right = widgets.VBox([title, _model, _n_points, auto,
                              _nugget, _c, _sill, _range,
                              save_button
                              ],
                             layout=dict(width="40%", border="2px solid lightgreen")
                             )
        
        # Estructura de la interfaz
        container = widgets.HBox([left, right], layout=dict(width="90%", height="400px"))
        
        # Mostrar el widget
        display(container)
        