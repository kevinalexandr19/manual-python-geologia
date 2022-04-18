import matplotlib.pyplot as plt
import mplstereonet
from ipywidgets import HTML, IntSlider, Button, HBox, VBox, interactive_output
from IPython.display import display, Markdown

plt.style.use("dark_background")


def plot_stereonet(rotation, strike, dip, rake):
    fig = plt.figure(figsize=(6, 6), constrained_layout=True)
    
    ax = fig.add_subplot(111, projection="stereonet", rotation=rotation)
    
    ax.plane(strike, dip, color="lightgreen", linewidth=2)
    ax.pole(strike, dip, color="red", ms=12)
    ax.rake(strike, dip, rake, color="gold", ms=10)

    ax.grid()
    
    plt.show()

     
def stereonet():     
    # Title
    title = """<div style="text-align: right"><font size="10">Stereonet Interactiva</font></div>"""
    title = HTML(title)
    title.layout.padding = "0% 0 10% 0%"
       
    # Stereonet sliders
    style = {"description_width": "30%"}
    layout = {"width": "95%"}
    features = ["""<font size="3">""" + x + """</font>""" for x in ["Rotación", "Rumbo", "Buzamiento", "Cabeceo"]]  
    
    rotation = IntSlider(min=0, max=360, step=5, value=0, description=features[0], layout=layout, style=style)
    strike = IntSlider(min=0, max=360, step=5, value=90, description=features[1], layout=layout, style=style)
    dip = IntSlider(min=0, max=90, step=1, value=45, description=features[2], layout=layout, style=style)
    rake = IntSlider(min=-90, max=90, step=1, value=90, description=features[3], layout=layout, style=style)
        
    # Reset button
    reset = Button(description="", button_style="", tooltip="", icon="fa-retweet", style={"button_color": "green"})
    reset_button = HBox([reset])
    reset_button.layout.width = "30%"
    reset_button.layout.padding = "2% 0 0 10%"
        
    def reset_click(b):
        rotation.value = 0
        strike.value = 90
        dip.value = 45
        rake.value = 90
        
    reset.on_click(reset_click)            
    
    # Container
    container = VBox([title, rotation, strike, dip, rake, reset_button])
    container.layout.border = "solid 1px green"
    container.layout.padding = "4% 3% 0% 0%"
    container.layout.width = "40%"
    container.layout.height = "580px"
    
    # Output
    output = interactive_output(plot_stereonet, {"rotation": rotation, "strike": strike, "dip": dip, "rake": rake})
    output.clear_output(wait = True)
    output.layout.border = "solid 1px green"
    output.layout.padding = "4% 4% 4% 4%"
    output.layout.width = "50%"
    output.layout.height = "580px"  
  
    display(HBox([container, output]))