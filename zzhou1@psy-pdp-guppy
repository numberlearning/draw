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

from analysis import read_img, write_img, T, read_n

clear_output()
b = Button(description="Loading...", icon="arrow", width=400)
dropdown = Dropdown(
    options=['0', '1000', '2000', '3000', '4000', '5000', '6000', '7000', '8000', '9000', '10000', '11000', '12000', '13000', '14000', '15000', '100000', '200000', '300000', '400000', '490000', '3474000'], 
    value='3474000',
    description='Iteration:'
)

action_dropdown = Dropdown(
    options=['read', 'write'],
    value='read',
    description='Action:'
)

figures = list()
iqs = list()


def make_figure(color, i):
    """
    Make the figure p with the image iii and attention window q.
    color: attention window color
    i: glimpse number
    """

    w = 100#28
    name = "Draw"
    title = "%s Glimpse %d" % (name, (i + 1))
    p = figure(x_range=(0, w), y_range=(w, 0), width=200, height=200, tools="", title=title, background_fill_color="#111111")
    
    p.toolbar.logo = None
    p.toolbar_location = None
    p.axis.visible = False
    p.border_fill_color = "#111111"
    p.title.text_color = "#DDDDDD"

    im = np.zeros((w, w))
    i_source = ColumnDataSource(data=dict(image=[im]))

    iii = p.image(image=[im], x=0, y=w, dw=w, dh=w, palette="Greys256")#"Spectral9")#"Greys256")
    source = ColumnDataSource(data=dict(top=[0], bottom=[0], left=[0], right=[0]))
    q = p.quad('left', 'right', 'top', 'bottom', source=source, color=color, fill_alpha=0, line_width=3)
    
        
    callback = CustomJS(code="""
    console.log(cb_data);
    if (IPython.notebook.kernel !== undefined) {
        var kernel = IPython.notebook.kernel;
        var i = %d;
        if (!this.hovered) {
            cmd = "hover(" + i + ")";
            kernel.execute(cmd, {}, {});
            this.hovered = true;
        }
        
        var that = this;
        
        
        document.querySelectorAll(".bk-plot-layout.bk-layout-fixed").forEach(function(x) {
            x.onmouseleave = function() {
                if (!that.hovered) {
                    return;
                }
                that.hovered = false;
                cmd = "unhover(" + i + ")";
                kernel.execute(cmd, {}, {});
            }
        })
        
    }
    """ % i)
    #p.add_tools(HoverTool(tooltips=None, callback=callback, renderers=[iii, q]))
    
    return p, iii, q;


for i in range(T):
    if True:#i % 2 == 0:
        (p1, i1, q1) = make_figure("pink", i)
        figures.append(p1)
        iqs.append((i1, q1))

        
data = None
    

def hover(i):
    """
    Show attention window image when figure is hovered over.
    i: glimpse number
    iqs: list of images and attention windows
    """

    iqs[i][0].data_source.data["image"][0] = data["rs"][i]
    iqs[i][1].data_source.data = dict(top=[0], bottom=[0], left=[0], right=[0])
    push_notebook(handle=handle)


def unhover(i):
    """
    Show figure image when figure is unhovered.
    i: glimpse number
    iqs: list of images and attention windows
    """

    iqs[i][0].data_source.data["image"][0] = data["img"]
    iqs[i][1].data_source.data = data["rects"][i]
    push_notebook(handle=handle)
    
    
def update_figures(handle, new_image=True):
    """Display figures at new iteration number."""

    global data
    if action_dropdown.value is 'read':
        data = read_img(int(dropdown.value), new_image)
    #    print(data["rects"])

        for i, f in enumerate(figures):
            picture = f
            picture_i, picture_q = iqs[i]
            picture_i.data_source.data["image"][0] = data["img"]
            picture_q.data_source.data = data["rects"][i]

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
