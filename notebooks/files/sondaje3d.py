####
#### SONDAJE3D v1.0 Mayo 2023
#### Módulo para visualizar información de sondajes en 3D
#### Nota: es importante limpiar y validar los datos previamente para evitar problemas en la visualización
#### Requires: numpy, pandas, numba, ipywidgets, scipy, tqdm, plotly

#### Copyright 2022, kevinalexandr19
#### This software may be modified and distributed under the terms of the MIT license.
#### See the LICENSE file for details.


import pandas as pd
import numpy as np
from numba import jit
import ipywidgets as widgets
from scipy import interpolate
from tqdm import tqdm
import plotly.express as px
import plotly.graph_objects as go


@jit
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
    collar : pandas.DataFrame, contains the drillhole collar data. You must select the columns: ID, X, Y, Z.\n
    survey : pandas.DataFrame, contains the drillhole survey data. You must select the columns: ID, AT, AZ, DIP.\n
    table: pandas.DataFrame, contains the drillhole geological data. You must select the columns: ID, FROM, TO.
           The remaining columns are used as features for the 3D visualization (e.g. AU_gpt).\n"""        
        
    def __init__(self, collar: pd.DataFrame, survey: pd.DataFrame, table: pd.DataFrame):
        self.collar = collar.copy()
        self.survey = survey.copy()
        self.table = table.copy()
        self.dtypes = None
        self.feature_points = None
        
        self.select_columns()
    
    
    def select_columns(self):
        """Widget that helps select each column for collar, survey and
           table. After the columns are selected, pressing the button
           will process the data and extract the datatype of each
           column in table (except from ID, FROM and TO)."""
        
        collar = self.collar
        survey = self.survey
        table = self.table

        ################ COLLAR ################
        # Lista de columnas a elegir
        optionsC = list(collar.columns) + ["Seleccione una columna"]

        # Crear widgets de selección para las columnas
        col_widget_1C = widgets.Dropdown(options=optionsC, value=optionsC[-1],
                                         description="", layout=widgets.Layout(width="200px"))
        col_widget_2C = widgets.Dropdown(options=optionsC, value=optionsC[-1],
                                         description="", layout=widgets.Layout(width="200px"))
        col_widget_3C = widgets.Dropdown(options=optionsC, value=optionsC[-1],
                                         description="", layout=widgets.Layout(width="200px"))
        col_widget_4C = widgets.Dropdown(options=optionsC, value=optionsC[-1],
                                         description="", layout=widgets.Layout(width="200px"))

        # Crear widgets de texto para los nombres fijos
        name_widget_1C = widgets.HTML(value="<p align='center'><b>ID</b></p>", layout=widgets.Layout(border="solid 1px green"))
        name_widget_2C = widgets.HTML(value="<p align='center'><b>X</b></p>", layout=widgets.Layout(border="solid 1px green"))
        name_widget_3C = widgets.HTML(value="<p align='center'><b>Y</b></p>", layout=widgets.Layout(border="solid 1px green"))
        name_widget_4C = widgets.HTML(value="<p align='center'><b>Z</b></p>", layout=widgets.Layout(border="solid 1px green"))

        # Mostrar los widgets
        display(widgets.VBox([widgets.Label("Seleccione las columnas de COLLAR: ID, X, Y y Z:"),
                              widgets.HBox([widgets.VBox([name_widget_1C, col_widget_1C]),
                                            widgets.VBox([name_widget_2C, col_widget_2C]),
                                            widgets.VBox([name_widget_3C, col_widget_3C]),
                                            widgets.VBox([name_widget_4C, col_widget_4C])
                                           ])
                             ]))

        ################ SURVEY ################
        # Lista de columnas a elegir
        optionsS = list(survey.columns) + ["Seleccione una columna"]

        # Crear widgets de selección para las columnas
        col_widget_1S = widgets.Dropdown(options=optionsS, value=optionsS[-1],
                                         description="", layout=widgets.Layout(width="200px"))
        col_widget_2S = widgets.Dropdown(options=optionsS, value=optionsS[-1],
                                         description="", layout=widgets.Layout(width="200px"))
        col_widget_3S = widgets.Dropdown(options=optionsS, value=optionsS[-1],
                                         description="", layout=widgets.Layout(width="200px"))
        col_widget_4S = widgets.Dropdown(options=optionsS, value=optionsS[-1],
                                         description="", layout=widgets.Layout(width="200px"))

        # Crear widgets de texto para los nombres fijos
        name_widget_1S = widgets.HTML(value="<p align='center'><b>ID</b></p>", layout=widgets.Layout(border="solid 1px green"))
        name_widget_2S = widgets.HTML(value="<p align='center'><b>AT</b></p>", layout=widgets.Layout(border="solid 1px green"))
        name_widget_3S = widgets.HTML(value="<p align='center'><b>AZ</b></p>", layout=widgets.Layout(border="solid 1px green"))
        name_widget_4S = widgets.HTML(value="<p align='center'><b>DIP</b></p>", layout=widgets.Layout(border="solid 1px green"))

        # Mostrar los widgets
        display(widgets.VBox([widgets.Label("Seleccione las columnas de SURVEY: ID, AT, AZ y DIP:"),
                              widgets.HBox([widgets.VBox([name_widget_1S, col_widget_1S]),
                                            widgets.VBox([name_widget_2S, col_widget_2S]),
                                            widgets.VBox([name_widget_3S, col_widget_3S]),
                                            widgets.VBox([name_widget_4S, col_widget_4S])
                                           ])
                             ]))

        ################ TABLE ################
        # Lista de columnas a elegir
        optionsT = list(table.columns) + ["Seleccione una columna"]

        # Crear widgets de selección para las columnas
        col_widget_1T = widgets.Dropdown(options=optionsT, value=optionsT[-1],
                                         description="", layout=widgets.Layout(width="200px"))
        col_widget_2T = widgets.Dropdown(options=optionsT, value=optionsT[-1],
                                         description="", layout=widgets.Layout(width="200px"))
        col_widget_3T = widgets.Dropdown(options=optionsT, value=optionsT[-1],
                                         description="", layout=widgets.Layout(width="200px"))

        # Crear widgets de texto para los nombres fijos
        name_widget_1T = widgets.HTML(value="<p align='center'><b>ID</b></p>", layout=widgets.Layout(border="solid 1px green"))
        name_widget_2T = widgets.HTML(value="<p align='center'><b>FROM</b></p>", layout=widgets.Layout(border="solid 1px green"))
        name_widget_3T = widgets.HTML(value="<p align='center'><b>TO</b></p>", layout=widgets.Layout(border="solid 1px green"))   

        #######################################
        # Output que muestra los nombres actuales de las columnas
        output = widgets.Output()

        # Crear un widget de botón para realizar el cambio de nombres
        button = widgets.Button(description="Asignar las columnas seleccionadas",
                                layout=widgets.Layout(width="800px", background="green"))
        button.style.button_color = "green"

        # Definir una función de manejo de eventos para el botón
        def on_button_click(b):
            columnsC = {col_widget_1C.value : "ID",
                        col_widget_2C.value : "X",
                        col_widget_3C.value : "Y",
                        col_widget_4C.value : "Z"}

            columnsS = {col_widget_1S.value : "ID",
                        col_widget_2S.value : "AT",
                        col_widget_3S.value : "AZ",
                        col_widget_4S.value : "DIP"}

            columnsT = {col_widget_1T.value : "ID",
                        col_widget_2T.value : "FROM",
                        col_widget_3T.value : "TO"}

            collar.rename(columns=columnsC, inplace=True)
            survey.rename(columns=columnsS, inplace=True)
            table.rename(columns=columnsT, inplace=True)

            with output:
                validate_collar = all([col in collar.columns for col in ["ID", "X", "Y", "Z"]])
                validate_survey = all([col in survey.columns for col in ["ID", "AT", "AZ", "DIP"]])
                validate_table = all([col in table.columns for col in ["ID", "FROM", "TO"]])

                if all([validate_collar, validate_survey, validate_table]):
                    print("Todas las columnas han sido seleccionadas.")
                    self.collar = collar[["ID", "X", "Y", "Z"]].copy()
                    self.survey = survey[["ID", "AT", "AZ", "DIP"]].copy()
                    
                    # Procesa toda la información
                    self.validate_columns()
                    self.validate_survey()
                    self.validate_ID()
                    self.get_table_points()
                    
                else:
                    print("Error: no todas las columnas han sido seleccionadas, reinicie el proceso.")

                output.clear_output(wait=True)

        # Asignar la función de manejo de eventos al botón
        button.on_click(on_button_click)

        # Mostrar los widgets
        display(widgets.VBox([widgets.Label("Seleccione las columnas de TABLE: ID, FROM y TO:"),
                              widgets.HBox([widgets.VBox([name_widget_1T, col_widget_1T]),
                                            widgets.VBox([name_widget_2T, col_widget_2T]),
                                            widgets.VBox([name_widget_3T, col_widget_3T]),
                                           ])
                             ]))

        display(widgets.VBox([button, output]))
        
    
    
    def validate_columns(self):
        """Validates the datatype of each column in collar, survey and table (except for features)."""
        
        # Validate and transform the datatypes in collar ID, X, Y, Z
        self.collar["ID"] = self.collar["ID"].astype(str)
        self.collar["X"] = pd.to_numeric(self.collar["X"], errors="coerce")
        self.collar["Y"] = pd.to_numeric(self.collar["Y"], errors="coerce")
        self.collar["Z"] = pd.to_numeric(self.collar["Z"], errors="coerce")
        
        # Validate and transform the datatypes in survey ID AT AZ DIP
        self.survey["ID"] = self.survey["ID"].astype(str)
        self.survey["AT"] = pd.to_numeric(self.survey["AT"], errors="coerce")
        self.survey["AT"] = np.where(self.survey["AT"] < 0, np.nan, self.survey["AT"])      # AT >= 0
        self.survey["AZ"] = pd.to_numeric(self.survey["AZ"], errors="coerce")
        self.survey["AZ"] = np.where(self.survey["AZ"] < 0, np.nan, self.survey["AZ"])      # AZ >= 0
        self.survey["AZ"] = np.where(self.survey["AZ"] > 360, np.nan, self.survey["AZ"])    # AZ <= 360
        self.survey["DIP"] = pd.to_numeric(self.survey["DIP"], errors="coerce")
        self.survey["DIP"] = np.where(self.survey["DIP"] <= 0, np.nan, self.survey["DIP"])  # DIP > 0
        self.survey["DIP"] = np.where(self.survey["DIP"] >= 90, np.nan, self.survey["DIP"]) # DIP < 90
    
        # Validate and transform the datatypes in table ID FROM TO
        self.table["ID"] = self.table["ID"].astype(str)
        self.table["FROM"] = pd.to_numeric(self.table["FROM"], errors="coerce")
        self.table["FROM"] = np.where(self.table["FROM"] < 0, np.nan, self.table["FROM"])   # FROM >= 0
        self.table["TO"] = pd.to_numeric(self.table["TO"], errors="coerce")
        self.table["TO"] = np.where(self.table["TO"] < 0, np.nan, self.table["TO"])         # TO >= 0
        
        print("Las columnas han sido validadas.")
                    
    
    def validate_survey(self):
        """Validates that each drillhole has more than one survey row.
           The drillholes with one or zero survey rows are deleted."""        
        
        collar = self.collar
        survey = self.survey
        table = self.table
        one_row_dhs = []
        
        for dh in survey["ID"].unique():
            count = len(survey[survey["ID"] == dh])
            if count <= 1:
                one_row_dhs.append(dh)
            else:
                pass
        
        if len(one_row_dhs) > 0:
            for dh in one_row_dhs:
                collar.drop(collar.loc[collar["ID"] == dh].index, inplace=True)
                survey.drop(survey.loc[survey["ID"] == dh].index, inplace=True)
                table.drop(table.loc[table["ID"] == dh].index, inplace=True)
        else:
            pass
            
        print("Las columnas de survey han sido revisadas.")
            
            
    def validate_ID(self):
        """Validates if there are no extra drillholes in some of the tables.
           If there are, they are deleted from the data."""  
        
        collar = self.collar
        survey = self.survey
        table = self.table        
        
        c = set(collar["ID"].unique())
        s = set(survey["ID"].unique())
        t = set(table["ID"].unique())
        extra_dhs = c ^ s | t ^ s | c ^ t
        
        if len(extra_dhs) > 0:
            for dh in extra_dhs:
                collar.drop(collar.loc[collar["ID"] == dh].index, inplace=True)
                survey.drop(survey.loc[survey["ID"] == dh].index, inplace=True)
                table.drop(table.loc[table["ID"] == dh].index, inplace=True)
        else:
            pass
        
        print("El ID de cada taladro ha sido revisado.")
    
    
    def set_table_datatypes(self):
        """Sets the datatype of each column in table except from ID, FROM, TO.
           For example, if AU_gpt contains float values, it will be assigned 
           as a numeric column."""
        
        table = self.table
        
        table_cols = list(table.columns)
        table_cols.remove("ID")
        table_cols.remove("FROM")
        table_cols.remove("TO")
        
        dtypes = dict()
        for col in table_cols:
            col_numeric = pd.to_numeric(table[col], errors="coerce")
            col_na = col_numeric.isna().sum()
            
            if len(col_numeric) == col_na:
                dtypes[col] = "categoric"
            else:
                dtypes[col] = "numeric"
                table[col] = np.where(col_numeric >= 0, col_numeric, np.nan)
                
        return dtypes
    
    
    def get_points(self, feature: str, dtype):
        """Process one table column (depending of datatype) and generates 
           the points needed to visualize it with the 3D plot."""
        
        collar = self.collar
        survey = self.survey
        table = self.table[["ID", "FROM", "TO", feature]].copy()
        
        points = []
        holeid = survey["ID"].unique()

        for dh in holeid:
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
        
            if dtype == "categoric":    
                for a, b, col  in zip(dfrom, dto, column):
                    p1 = [float(i) for i in interpolate.splev(a, tck)]
                    p1.append(col)
                    p2 = [float(i) for i in interpolate.splev(b, tck)]
                    p2.append(col)
                    p3 = [None, None, None, col]
                    points.extend([p1, p2, p3])      
                    
            elif dtype == "numeric":
                for a, b, col in zip(dfrom, dto, column):
                    p1 = [float(i) for i in interpolate.splev(a, tck)]
                    p1.append(col)
                    p2 = [float(i) for i in interpolate.splev(b, tck)]
                    p2.append(col)
                    points.extend([p1, p2])
                
                points.append([None, None, None, np.nan])
        
        # La información procesada se almacena en el atributo points
        points = pd.DataFrame(points, columns=["X", "Y", "Z", feature])
        return points
    
    
    def get_table_points(self):
        """Process all the columns in table (except for ID, FROM, TO), in order
           to obtain the points needed to visualize it with the 3D plot."""
        
        dtypes = self.set_table_datatypes()
        feature_points = dict()
        
        print("Generando el input para la visualización 3D:")
        for col, dtype in tqdm(dtypes.items()):
            points = self.get_points(col, dtype)
            feature_points[col] = points
        
        self.dtypes = dtypes
        self.feature_points = feature_points
    
    
    def plot_3d(self, feature):
        """Plotly 3D visualization of drillholes. Numeric data is plotted using a continuous colorbar,
           categorical data is plotted using a discrete combination of colors."""
        
        collar = self.collar
        points = self.feature_points[feature]
        dtype = self.dtypes[feature]
        
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
        
        if dtype == "categoric":
            for value in points[feature].unique():
                df = points[points[feature] == value]
                fig.add_trace(go.Scatter3d(x=df["X"], y=df["Y"], z=df["Z"], name=value,
                                           legendgroup="feature", legendgrouptitle_text=feature,
                                           mode="lines", connectgaps=False,
                                           line=dict(width=7)
                                          )
                             )
        elif dtype == "numeric":
            fig.add_trace(go.Scatter3d(x=points["X"], y=points["Y"], z=points["Z"], name=feature, 
                                       legendgroup="feature", legendgrouptitle_text=feature,
                                       mode="lines", connectgaps=False,
                                       line=dict(colorbar=dict(title=feature, x=-0.1,
                                                               titlefont=dict(color="white"),
                                                               tickfont=dict(color="white")
                                                              ),
                                                 width=7,
                                                 color=points[feature],
                                                 colorscale="Jet"
                                                )
                                      )
                         )
        
        fig.update_layout(autosize=False,
                          legend = dict(bgcolor="white", itemsizing="constant", groupclick="toggleitem"),
                          width=900,
                          height=500,
                          margin=dict(l=50, r=50, b=50, t=50, pad=4),
                          paper_bgcolor="rgba(1, 1, 1, 1)",
                          plot_bgcolor="rgba(1, 1, 1, 1)",
                          scene=dict(xaxis_title="X",
                                     xaxis=dict(range=[xmin, xmax], backgroundcolor="black", color="white"),
                                     yaxis_title="Y",
                                     yaxis=dict(range=[ymin, ymax], backgroundcolor="black", color="white"),
                                     zaxis_title="Z",
                                     zaxis=dict(range=[zmin, zmax], backgroundcolor="black", color="white"),
                                     bgcolor="black"
                                    ),
                         )
        
        fig.show()

        
    def interactive_plot3d(self):
        """Interactive 3D visualization using Jupyter widgets."""
        
        output = widgets.Output()
        output.layout.border = "solid 1px green"

        with output:
            widgets.interact(self.plot_3d, feature=list(self.dtypes.keys()));
        
        display(output)



        