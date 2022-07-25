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
    collar : pandas.DataFrame, contains the drillhole collar data. Columns: BHID, XCOLLAR, YCOLLAR, ZCOLLAR.\n
    survey : pandas.DataFrame, contains the drillhole survey data. Columns: BHID, AT, AZ, DIP.\n
    table: pandas.DataFrame, contains the drillhole geological data. Columns: BHID, FROM, TO, and any combination of features.\n"""        
        
    def __init__(self, collar: pd.DataFrame, survey: pd.DataFrame, table: pd.DataFrame, table_name: str):
        print("Los sondajes deben ser validados antes de realizar la visualización 3D.")
        self.collar = collar.copy()
        self.survey = survey.copy()
        self.table = table.copy()
        self.table_name = table_name
        self.validated = False
        self.has_points = False
        # Falta validar collar.unique == survey.unique 
        
        
    def validate_columns(self, df: pd.DataFrame, name: str, columns: list, istable=False):
        """Validates the name of each column in collar, survey and table."""
        
        print(f"    Validación de columnas en {name}:")
        dfcols = list(df.columns)
        count = 0
        
        for col in dfcols:
            if col in columns:
                print(f"        Columna {col}: incluida")
            else:
                if istable:
                    print(f"        Columna adicional: {col}")
                else:
                    count += 1
                    print(f"        Columna adicional {col}: es necesario remover esta columna")
        
        if count == 0:
            self.validated = True
    
    
    def validate_datatypes(self, df, name, dtypes: dict, istable=False):       
        """Validates the data type of each column in collar, survey and table."""
        
        print(f"    Validación de tipos de datos en {name}:")
        df_dtypes = dict(df.dtypes)
        count = 0
        
        for col, dtype in dtypes.items():
            if df_dtypes[col] == dtype:
                print(f"        Columna {col}: {dtype}")
            else:
                count += 1
                print(f"        Columna {col}: tipo de dato incorrecto. Cambiar a {dtype}")
        
        if count == 0:
            self.validated = True
        else:
            self.validated = False
    
    
    def validate_survey(self):
        """Validates that each drillhole has more than one survey row"""
        survey = self.survey
        one_row_dh = []
        
        for dh in survey["BHID"].unique():
            count = len(survey[survey["BHID"] == dh])
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
            
            
    def validate_BHID(self):
        """Validates if there are no extra drillholes in some of the tables."""
        c = set(self.collar["BHID"].unique())
        s = set(self.survey["BHID"].unique())
        t = set(self.table["BHID"].unique())
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
        print("Validación de información en collar:")
        self.validate_columns(self.collar, "collar", ["BHID", "XCOLLAR", "YCOLLAR", "ZCOLLAR"])
        self.validate_datatypes(self.collar, "collar", {"BHID": "object", "XCOLLAR": "float64", "YCOLLAR": "float64", "ZCOLLAR": "float64"})
        print("")
        
        print("Validación de información en survey:")
        self.validate_columns(self.survey, "survey", ["BHID", "AT", "AZ", "DIP"])
        self.validate_datatypes(self.survey, "survey", {"BHID": "object", "AT": "float64", "AZ": "float64", "DIP": "float64"})
        print("")
        
        print(f"Validación de información en {self.table_name}:")
        self.validate_columns(self.table, self.table_name, ["BHID", "FROM", "TO"], istable=True)
        self.validate_datatypes(self.table, self.table_name, {"BHID": "object", "FROM": "float64", "TO": "float64"})
        print("")
        
        print(f"Validación de registros en survey por cada sondaje")
        self.validate_survey()
        print("")
        
        print(f"Validación de sondajes adicionales en algunos archivos:")
        self.validate_BHID()
        print("")
        
        if self.validated == True:
            print("Los sondajes han sido validados.")
        else:
            print("La validación de los sondajes ha fallado.")
    
    
    def get_points(self, feature: str):
        if self.validated == False:
            return print("La información debe ser validada primero a través del método validate.")
        
        self.feature = feature
        collar = self.collar
        survey = self.survey
        table = self.table
        
        points = []
        table = self.table[["BHID", "FROM", "TO", feature]].copy()
        bhid = self.survey["BHID"].unique()
        
        print(f"Procesando la información de {len(bhid)} sondajes. Columna: {feature}")
        for dh in tqdm(bhid):
            # Información del sondaje
            dh_collar = collar[collar["BHID"] == dh].values[0][1:].astype(float)
            dh_survey = survey[survey["BHID"] == dh].values[:, 1:].astype(float)
            dh_feature = table[table["BHID"] == dh].values[:, 1:]
    
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
            col = dh_feature[:, 2]
    
            for u, v, r in zip(dfrom, dto, col):
                p1 = [float(i) for i in interpolate.splev(u, tck)]
                p1.append(r)
                p2 = [float(i) for i in interpolate.splev(v, tck)]
                p2.append(r)
                p3 = [None, None, None, r]
                points.extend([p1, p2, p3])
        
        # La información procesada se almacena en el atributo points
        self.points = pd.DataFrame(points, columns=["X", "Y", "Z", feature])
        self.has_points = True        
    
    
    def plot_3d(self):
        if self.has_points == False:
            return print("Los puntos para la visualización aún no han sido generados a través del método get_points.")
        
        collar = self.collar
        feature = self.feature
        points = self.points
        
        # Dimensiones del gráfico 3D
        xmin, xmax = round(collar["XCOLLAR"].min(), -3) - 1000, round(collar["XCOLLAR"].max(), -3) + 1000
        ymin, ymax = round(collar["YCOLLAR"].min(), -3) - 1000, round(collar["YCOLLAR"].max(), -3) + 1000
        zmin, zmax = round(collar["ZCOLLAR"].min(), -3) - 1000, round(collar["ZCOLLAR"].max(), -3) + 1000
        
        # Visualización 3D en Plotly
        fig = px.scatter_3d(data_frame=collar, x="XCOLLAR", y="YCOLLAR", z="ZCOLLAR", text="BHID")
        fig.update_traces(marker=dict(size=1, color="white"), textfont=dict(size=7, color="white"))
        
        for value in points[feature].unique():
            df = points[points[feature] == value]
            fig.add_trace(go.Scatter3d(x=df["X"], y=df["Y"], z=df["Z"], name=value,
                                       mode="lines", line=dict(width=7), connectgaps=False))
        
        fig.update_layout(
            autosize=False,
            legend = dict(bgcolor="white", title=f"{feature}", itemsizing="constant"),
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
            