####
#### SONDAJE 3D
#### Módulo para visualizar información de sondajes en 3D
#### Nota: es importante validar los datos previamente para evitar errores en la visualización

#### Copyright 2022, kevinalexandr19
#### This software may be modified and distributed under the terms of the MIT license.
#### See the LICENSE file for details.


import pandas as pd
import numpy as np
from scipy import interpolate
from tqdm import tqdm
import plotly.express as px
import plotly.graph_objects as go


def xyz(d, az1, dip1, az2, dip2):
    """     Interpolates the azimuth and dip angle over a line.\n
    Parameters\n    ----------
    d : units of distance between the two points.\n
    az1, dip1, az2, dip2 : float, azimuth and dip (downward positive) of both points.\n"""  
    
    az1 = np.radians(az1)
    dip1 = np.radians(90 - dip1)
    
    az2 = np.radians(az2)
    dip2 = np.radians(90 - dip2)
    
    dl = np.arccos(np.cos(dip2 - dip1) - np.sin(dip1)*np.sin(dip2)*(1 - np.cos(az2 - az1)))
    
    if dl != 0:
        rf = 2*np.tan(dl/2)/dl  # minimun curvature
    else:
        rf = 1                  # balanced tangential
        
    dz = -0.5*d*(np.cos(dip1) + np.cos(dip2))*rf
    dy = 0.5*d*(np.sin(dip1)*np.cos(az1) + np.sin(dip2)*np.cos(az2))*rf
    dx = 0.5*d*(np.sin(dip1)*np.sin(az1) + np.sin(dip2)*np.sin(az2))*rf
    
    return np.array([dx, dy, dz])


class DrillData:
    """    Drillhole data containing collar, survey and assay information.\n
    Parameters\n    ----------
    collar : pandas.DataFrame, contains the drillhole collar data. Columns: ID, X, Y, Z.\n
    survey : pandas.DataFrame, contains the drillhole survey data. Columns: ID, AT, AZ, DIP.\n
    table: pandas.DataFrame, contains the drillhole geological data. Columns: ID, FROM, TO, and any combination of features.\n"""        
        
    def __init__(self, collar: pd.DataFrame, survey: pd.DataFrame, table: pd.DataFrame, table_name: str):
        print("Los sondajes deben ser validados antes de realizar la visualización 3D.")
        self.collar = collar.copy()
        self.survey = survey.copy()
        self.table = table.copy()
        self.table_name = table_name
        self.validated = False
        self.has_points = False
        self.feature = None
        self.feature_points = None
        self.trajectory = None
        
        
    def validate_columns(self, df: pd.DataFrame, name: str, columns: list, istable=False):
        """Validates the name of each column in collar, survey and table."""
        
        print(f"\033[1m    Validación de columnas en {name}:\033[0m")
        dfcols = list(df.columns)
        count = len(columns)
        
        missing_cols = columns
        for col in dfcols:
            if col in columns:
                print(f"\033[1m        Columna {col}: incluida\033[0m")
                missing_cols.remove(col)
                count -= 1
            else:
                print(f"        Columna adicional: {col}")
        
        if count == 0:
            self.validated = True
        else:
            print(f"    Faltan las siguientes columnas en {name}: {missing_cols}")
            if istable == False:
                print(f"    Existen {count} columnas adicionales en {name}, renombrar o remover estas columnas para resolver el problema.")
    
    
    def validate_datatypes(self, df, name, dtypes: dict, istable=False):
        """Validates the data type of each column in collar, survey and table."""
        
        if not self.validated:
            return print("\033[1m    Las columnas no han sido validadas. No es posible validar los tipos de datos.\033[0m")
        
        print(f"\033[1m    Validación de tipos de datos en {name}:\033[0m")
        df_dtypes = dict(df.dtypes)
        count = 0
        
        for col, dtype in dtypes.items():
            if df_dtypes[col] == dtype:
                print(f"\033[1m        Columna {col}: {dtype}\033[0m")
            else:
                count += 1
                print(f"        Columna {col}: tipo de dato incorrecto. Cambiar a {dtype}")
        
        if count == 0:
            self.validated = True
        else:
            self.validated = False
    
    
    def validate_survey(self):
        """Validates that each drillhole has more than one survey row"""
        
        if not self.validated:
            return print("\033[1m    Las columnas no han sido validadas. No es posible validar el contenido de survey.\033[0m")        
        
        survey = self.survey
        one_row_dh = []
        
        for dh in survey["ID"].unique():
            count = len(survey[survey["ID"] == dh])
            if count <= 1:
                one_row_dh.append(dh)
                print(f"    Remover sondaje {dh} con data insuficiente en survey")
            else:
                pass
        
        if len(one_row_dh) == 0:
            print("    Todos los sondajes tienen más de una entrada en survey.")
            self.validated = True
        else:
            self.validated = False    
            
            
    def validate_ID(self):
        """Validates if there are no extra drillholes in some of the tables."""
        
        if not self.validated:
            return print("\033[1m    Las columnas no han sido validadas. No es posible validar el ID de los sondajes.\033[0m")   
        
        c = set(self.collar["ID"].unique())
        s = set(self.survey["ID"].unique())
        t = set(self.table["ID"].unique())
        extra_dh = c ^ s | t ^ s | c ^ t
        
        if len(extra_dh) == 0:
            print("    Todos los archivos contienen los mismos sondajes.")
            self.validated = True
        else:
            print("    Algunos archivos contienen sondajes que no aparecen en otros:")
            for dh in extra_dh:
                if dh in c:
                    print(f"        Remover sondaje {dh} en collar")
                else:
                    pass
            for dh in extra_dh:
                if dh in s:
                    print(f"        Remover sondaje {dh} en survey")
                else:
                    pass
            for dh in extra_dh:
                if dh in t:
                    print(f"        Remover sondaje {dh} en {self.table_name}")
                else: pass
            self.validated = False       
    
        
    def validate(self):
        print("\033[1mValidación de información en collar:\033[0m")
        self.validate_columns(self.collar, "collar", ["ID", "X", "Y", "Z"])
        self.validate_datatypes(self.collar, "collar", {"ID": "object", "X": "float64", "Y": "float64", "Z": "float64"})
        print("")
        
        print("\033[1mValidación de información en survey:\033[0m")
        self.validate_columns(self.survey, "survey", ["ID", "AT", "AZ", "DIP"])
        self.validate_datatypes(self.survey, "survey", {"ID": "object", "AT": "float64", "AZ": "float64", "DIP": "float64"})
        print("")
        
        print(f"\033[1mValidación de información en {self.table_name}:\033[0m")
        self.validate_columns(self.table, self.table_name, ["ID", "FROM", "TO"], istable=True)
        self.validate_datatypes(self.table, self.table_name, {"ID": "object", "FROM": "float64", "TO": "float64"})
        print("")
        
        print(f"\033[1mValidación de registros en survey por cada sondaje\033[0m")
        self.validate_survey()
        print("")
        
        print(f"\033[1mValidación de sondajes adicionales en algunos archivos:\033[0m")
        self.validate_ID()
        print("")
        
        if self.validated == True:
            print("\033[1mLos sondajes han sido validados.\033[0m")
        else:
            print("\033[1mLa validación de los sondajes ha fallado.\033[0m")
    
    
    def get_points(self, feature: str, dtype: str):
        if self.validated == False:
            return print("\033[1mLa información debe ser validada primero a través del método validate.\033[0m")
        
        self.feature = feature
        self.feature_dtype = dtype
        collar = self.collar
        survey = self.survey
        table = self.table
        
        points = []
        table = self.table[["ID", "FROM", "TO", feature]].copy()
        holeid = self.survey["ID"].unique()

        print(f"Procesando la información de {len(holeid)} sondajes. Columna: {feature}")

        if dtype == "categoric":
            for dh in tqdm(holeid):
                # Información del sondaje
                dh_collar = collar[collar["ID"] == dh].values[0][1:].astype(float)
                dh_survey = survey[survey["ID"] == dh].values[:, 1:].astype(float)
                dh_feature = table[table["ID"] == dh].values[:, 1:]
    
                # Direcciones del sondaje
                lengths = dh_survey[1:, 0] - dh_survey[:-1, 0]
                knots = [np.array([0, 0, 0])]
                for d, array1, array2 in zip(lengths, dh_survey[:-1, 1:], dh_survey[1:, 1:]):
                    knots.append(xyz(d, array1[0], array1[1], array2[0], array2[1]))
                knots = np.cumsum(knots, axis=0)
    
                # Coordenadas del sondaje
                knots = knots + dh_collar
            
                # Interpolación de puntos en la dirección del sondaje
                if len(dh_survey) > 3:
                    tck, u = interpolate.splprep(knots.T, k=2)    
                else:
                    tck, u = interpolate.splprep(knots.T, k=1)
    
                # Interpolación de puntos a lo largo de todo el sondaje
                length = dh_feature[:, 1].max()
                dfrom = dh_feature[:, 0] / length
                dto = dh_feature[:, 1] / length
                column = dh_feature[:, 2]
    
                for a, b, col  in zip(dfrom, dto, column):
                    p1 = [float(i) for i in interpolate.splev(a, tck)]
                    p1.append(col)
                    p2 = [float(i) for i in interpolate.splev(b, tck)]
                    p2.append(col)
                    p3 = [None, None, None, col]
                    points.extend([p1, p2, p3])
        
        elif dtype == "numeric":
            for dh in tqdm(holeid):
                # Información del sondaje
                dh_collar = collar[collar["ID"] == dh].values[0][1:].astype(float)
                dh_survey = survey[survey["ID"] == dh].values[:, 1:].astype(float)
                dh_feature = table[table["ID"] == dh].values[:, 1:]
    
                # Direcciones del sondaje
                lengths = dh_survey[1:, 0] - dh_survey[:-1, 0]
                knots = [np.array([0, 0, 0])]
                for d, array1, array2 in zip(lengths, dh_survey[:-1, 1:], dh_survey[1:, 1:]):
                    knots.append(xyz(d, array1[0], array1[1], array2[0], array2[1]))
                knots = np.cumsum(knots, axis=0)
    
                # Coordenadas del sondaje
                knots = knots + dh_collar
            
                # Interpolación de puntos en la dirección del sondaje
                if len(dh_survey) > 3:
                    tck, u = interpolate.splprep(knots.T, k=2)    
                else:
                    tck, u = interpolate.splprep(knots.T, k=1)
    
                # Interpolación de puntos a lo largo de todo el sondaje
                length = dh_feature[:, 1].max()
                dfrom = dh_feature[:, 0] / length
                dto = dh_feature[:, 1] / length
                column = dh_feature[:, 2]
    
                for a, b, col in zip(dfrom, dto, column):
                    p1 = [float(i) for i in interpolate.splev(a, tck)]
                    p1.append(col)
                    p2 = [float(i) for i in interpolate.splev(b, tck)]
                    p2.append(col)
                    points.extend([p1, p2])
                
                points.append([None, None, None, np.nan])
        
        # La información procesada se almacena en el atributo points
        self.points = pd.DataFrame(points, columns=["X", "Y", "Z", feature])
        self.has_points = True
    
    
    def plot_3d(self):
        if self.has_points == False:
            return print("\033[1mLos puntos para la visualización aún no han sido generados a través del método get_points.\033[0m")
        
        collar = self.collar
        feature = self.feature
        points = self.points
        
        # Dimensiones del gráfico 3D
        xmin, xmax = round(collar["X"].min(), -3) - 1000, round(collar["X"].max(), -3) + 1000
        ymin, ymax = round(collar["Y"].min(), -3) - 1000, round(collar["Y"].max(), -3) + 1000
        zmin, zmax = round(collar["Z"].min(), -3) - 1000, round(collar["Z"].max(), -3) + 1000
        
        # Visualización 3D en Plotly
        fig = go.Figure()
        
        fig.add_trace(go.Scatter3d(x=collar["X"], y=collar["Y"], z=collar["Z"], text=collar["ID"], name="Collar",
                                   legendgroup="drillhole", legendgrouptitle_text="Drillhole",
                                   mode="markers+text", marker=dict(size=1, color="lightgray"), textfont=dict(size=7, color="white")))
        
        fig.add_trace(go.Scatter3d(x=points["X"], y=points["Y"], z=points["Z"], name="Path",
                                   legendgroup="drillhole", legendgrouptitle_text="Drillhole",
                                   mode="lines", line=dict(width=0.8, color="lightgray"), connectgaps=False))
        
        if self.feature_dtype == "categoric":
            for value in points[feature].unique():
                df = points[points[feature] == value]
                fig.add_trace(go.Scatter3d(x=df["X"], y=df["Y"], z=df["Z"], name=value,
                                           legendgroup="feature", legendgrouptitle_text=feature,
                                           mode="lines", line=dict(width=7), connectgaps=False))
        elif self.feature_dtype == "numeric":
            fig.add_trace(go.Scatter3d(x=points["X"], y=points["Y"], z=points["Z"], name=feature, 
                                       legendgroup="feature", legendgrouptitle_text=feature,
                                       mode="lines+text", connectgaps=False, 
                                       line=dict(colorbar=dict(title=feature, titlefont=dict(color="white"), x=-0.1, tickfont=dict(color="white")), 
                                                 colorscale="Jet", width=7, color=points[feature])
                                      )
                         )
        
        fig.update_layout(
            autosize=False,
            legend = dict(bgcolor="white", itemsizing="constant", groupclick="toggleitem"),
            width=1000,
            height=600,
            margin=dict(
                l=50,
                r=50,
                b=50,
                t=50,
                pad=4
            ),
            paper_bgcolor="rgba(1, 1, 1, 1)",
            plot_bgcolor="rgba(1, 1, 1, 1)",
            scene = dict(xaxis_title="X",
                         xaxis=dict(range=[xmin, xmax], backgroundcolor="black", color="white"),
                         yaxis_title="Y",
                         yaxis=dict(range=[ymin, ymax], backgroundcolor="black", color="white"),
                         zaxis_title="Z",
                         zaxis=dict(range=[zmin, zmax], backgroundcolor="black", color="white"),
                         bgcolor="black"
                         ),
                        )
        
        fig.show()
            