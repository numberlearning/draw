print("Setting everything up!")
import warnings
warnings.filterwarnings('ignore')

from bokeh.io import push_notebook, show, output_notebook
from bokeh.layouts import row, column
from bokeh.plotting import figure
from bokeh.models import ColumnDataSource, HoverTool, CustomJS, FixedTicker
import bokeh.palettes as pal
from bokeh.layouts import layout, Spacer, gridplot
output_notebook()

import ipywidgets as widgets
from ipywidgets import *
from IPython.display import display, clear_output

import numpy as np

from bokeh.charts import Bar, Histogram

from analysis import read_img, read_img2, write_img, T, read_n

clear_output()
b = Button(description="Loading...", icon="arrow", width=400)
dropdown = Dropdown(
    options=['0', '1000', '2000', '3000', '4000', '5000', '6000', '7000', '8000', '9000', '10000', '11000', '12000', '13000', '14000', '15000', '54000', '100000', '130000', '140000', '150000', '160000', '200000', '300000', '400000', '490000', '3474000'], 
    value='130000',
    description='Iteration:'
)

action_dropdown = Dropdown(
    options=['read', 'write'],
    value='read',
    description='Action:'
)

figures = list()
ids = list()


def make_figure(color, i):
    """
    Make the figure p with the image iii and attention window q.
    color: attention window color
    i: glimpse number
    """

    w = 28
    name = "Draw"
    title = "%s Glimpse %d" % (name, (i + 1))
    p = figure(x_range=(0, w), y_range=(w, 0), width=200, height=200, tools="", title=title, background_fill_color="#111111")
    
    p.toolbar.logo = None
    p.toolbar_location = None
    p.axis.visible = False
    p.border_fill_color = "#111111"
    p.title.text_color = "#DDDDDD"

    dots_source = ColumnDataSource(data=dict(mu_x_list=[0], mu_y_list=[0]))
    d = p.circle("mu_x_list", "mu_y_list", source=dots_source, size=5, color="orange", alpha=0.5)

    im = np.zeros((w, w))
    i_source = ColumnDataSource(data=dict(image=[im]))

    iii = p.image(image=[im], x=0, y=w, dw=w, dh=w, palette="Greys256")#"Spectral9")#"Greys256")

    return p, iii, d;


for i in range(T):
    if True:#i % 2 == 0:
        (p1, i1, d1) = make_figure("pink", i)
        figures.append(p1)
        ids.append((i1, d1))

        
data = None
    
    
def update_figures(handle, new_image=True):
    """Display figures at new iteration number."""

    global data
    if action_dropdown.value is 'read':
        data = read_img2(int(dropdown.value), new_image)
    #    print(data["rects"])

        for i, f in enumerate(figures):
            picture = f
            #picture_i, picture_q = iqs[i]
            picture_i, picture_d = ids[i]
            picture_i.data_source.data["image"][0] = data["img"]
            #picture_q.data_source.data = data["rects"][i]
            #print(data["dots"][i])
            picture_d.data_source.data = data["dots"][i]

    else:
        data = write_img(int(dropdown.value), new_image)
        #print(data["rects"])
        #print(data["c"])

        for i, f in enumerate(figures):
            picture = f
            picture_i, picture_q = iqs[i]
            picture_i.data_source.data["image"][0] = data["c"][i]
            picture_q.data_source.data = data["rects"][i]

    push_notebook(handle=handle)
    

def on_click(b, new_image=True):
    """Change figures after button is clicked."""

    b.description = "Loading..."
    update_figures(handle, new_image=new_image)
    b.description = "Next (Random) Image"

b.on_click(on_click)


def on_change(change):
    """Detect change of dropdown menu selection."""

    if change['type'] == 'change' and change['name'] == 'value':
        on_click(b, new_image=False)


dropdown.observe(on_change)
action_dropdown.observe(on_change)
display(HBox([b, dropdown, action_dropdown]))
handle = show(row(figures), notebook_handle=True)
on_click(b)
