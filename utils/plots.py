import plotly.graph_objs as go
import pandas as pd
import os
from ipywidgets import (Tab, SelectMultiple, Accordion, ToggleButton,
                        VBox, HBox, HTML, Image, Button, Text, Dropdown)
from ipywidgets import HBox, VBox, Image, Layout, HTML
import numpy as np
from scipy.spatial import ConvexHull
from sklearn.cluster import KMeans, DBSCAN
import json


class Figure1:
    def __init__(self, csv_file_path, base_path='.',
                 inchi_key='XFYICZOIWSBQSK-UHFFFAOYSA-N',
                 clustering=None):
        self.current_amine_inchi = inchi_key
        self.base_path = base_path
        self.clustering = clustering
        self.full_perovskite_data = pd.read_csv(
            csv_file_path, low_memory=False)
        # Filtering so that only 95 degree experiments are included
        self.full_perovskite_data = self.full_perovskite_data[self.full_perovskite_data['_raw_modelname'].str.contains('Uniform')]
        self.inchis = pd.read_csv('./data/inventory.csv')
        self.inchi_dict = dict(zip(self.inchis['Chemical Name'],
                                   self.inchis['InChI Key (ID)']))
        self.chem_dict = dict(zip(self.inchis['InChI Key (ID)'],
                                  self.inchis['Chemical Name']))
        # self.state_spaces = pd.read_csv('./perovskitedata/state_spaces.csv')
        self.ss_dict = json.load(open('./data/s_spaces.json', 'r'))
        self.generate_plot(self.current_amine_inchi)
        self.setup_widgets()

    def generate_plot(self, inchi_key):
        if inchi_key in self.ss_dict:
            self.setup_hull(hull_points=self.ss_dict[inchi_key])
        else:
            self.setup_hull()
        self.gen_amine_traces(inchi_key)
        self.setup_plot(yaxis_label=self.chem_dict[inchi_key]+' (M)')

    def setup_hull(self, hull_points=[[0., 0., 0.]]):
        xp, yp, zp = zip(*hull_points)
        self.hull_mesh = go.Mesh3d(x=xp,
                                   y=yp,
                                   z=zp,
                                   color='green',
                                   opacity=0.50,
                                   alphahull=0)

    def setup_success_hull(self, success_hull, success_points):
        if success_hull:
            xp, yp, zp = zip(*success_points[success_hull.vertices])
            self.success_hull_plot = go.Mesh3d(x=xp,
                                               y=yp,
                                               z=zp,
                                               color='red',
                                               opacity=0.50,
                                               alphahull=0)
        else:
            self.success_hull_plot = go.Mesh3d(x=[0],
                                               y=[0],
                                               z=[0],
                                               color='red',
                                               opacity=0.50,
                                               alphahull=0)

    def gen_amine_traces(self, inchi_key, amine_short_name='Me2NH2I'):
        amine_data = self.full_perovskite_data.loc[
            self.full_perovskite_data['_rxn_organic-inchikey']
            == inchi_key]

        success_hull = None

        # print(f'Total points: {len(amine_data)}')
        # print(self.ss_dict.keys())
        if inchi_key in self.ss_dict:
            xp, yp, zp = zip(*self.ss_dict[inchi_key])
        else:
            xp, yp, zp = [0], [0], [0]
        self.max_inorg = max([amine_data['_rxn_M_inorganic'].max(), max(xp)])
        self.max_org = max([amine_data['_rxn_M_organic'].max(), max(yp)])
        self.max_acid = max([amine_data['_rxn_M_acid'].max(), max(zp)])

        # Splitting by crystal scores. Assuming crystal scores from 1-4
        self.amine_crystal_dfs = []
        for i in range(1, 5):
            self.amine_crystal_dfs.append(
                amine_data.loc[amine_data['_out_crystalscore'] == i])
        # print(len(self.amine_crystal_dfs[3]))
        self.amine_crystal_traces = []
        self.trace_colors = ['rgba(65, 118, 244, 1.0)',
                             'rgba(92, 244, 65, 1.0)',
                             'rgba(244, 238, 66, 1.0)',
                             'rgba(244, 66, 66, 1.0)']

        for i, df in enumerate(self.amine_crystal_dfs):
            trace = go.Scatter3d(
                x=df['_rxn_M_inorganic'],
                y=df['_rxn_M_organic'],
                z=df['_rxn_M_acid'],
                mode='markers',
                name='Class {}'.format(i+1),
                text=["""<b>Lead Iodide [PbI2] (M)</b>: {:.3f} <br><b>{} (M)</b>: {:.3f}<br><b>Formic Acid [FAH] (M)</b>: {:.3f}""".format(
                    row['_rxn_M_inorganic'],
                    self.chem_dict[row['_rxn_organic-inchikey']],
                    row['_rxn_M_organic'],
                    row['_rxn_M_acid'])
                    for idx, row in df.iterrows()],
                hoverinfo='text',
                marker=dict(
                    size=4,
                    color=self.trace_colors[i],
                    line=dict(
                        width=0.2
                    ),
                    opacity=1.0
                )
            )
            self.amine_crystal_traces.append(trace)
            if i == 3:
                success_points = np.dstack((df['_rxn_M_inorganic'],
                                            df['_rxn_M_organic'],
                                            df['_rxn_M_acid']))[0]

                if len(success_points) > 3:
                    success_hull = ConvexHull(success_points)
                else:
                    success_hull = None

                self.setup_success_hull(success_hull, success_points)

        self.data = self.amine_crystal_traces

        self.data += [self.success_hull_plot]
        self.data += [self.hull_mesh]

        # if self.hull_mesh:

    def setup_plot(self, xaxis_label='Lead Iodide [PbI2] (M)',
                   yaxis_label='Dimethylammonium Iodide<br>[Me2NH2I] (M)',
                   zaxis_label='Formic Acid [FAH] (M)'):
        self.layout = go.Layout(
            scene=dict(
                xaxis=dict(
                    title=xaxis_label,
                    tickmode='linear',
                    dtick=0.5,
                    range=[0, self.max_inorg],
                ),
                yaxis=dict(
                    title=yaxis_label,
                    tickmode='linear',
                    dtick=0.5,
                    range=[0, self.max_org],
                ),
                zaxis=dict(
                    title=zaxis_label,
                    tickmode='linear',
                    dtick=1.0,
                    range=[0, self.max_acid],
                ),
            ),
            legend=go.layout.Legend(
                x=0,
                y=1,
                traceorder="normal",
                font=dict(
                    family="sans-serif",
                    size=12,
                    color="black"
                ),
                bgcolor="LightSteelBlue",
                bordercolor="Black",
                borderwidth=2
            ),
            width=975,
            height=700,
            margin=go.layout.Margin(
                l=20,
                r=20,
                b=20,
                t=20,
                pad=2
            ),
        )

        try:
            with self.fig.batch_update():
                for i, trace in enumerate(self.data):
                    self.fig.data[i].x = trace.x
                    self.fig.data[i].y = trace.y
                    self.fig.data[i].z = trace.z
                    self.fig.data[i].text = trace.text

                self.fig.layout.update(self.layout)
        except Exception as e:
            print(f'Exception: {e}')
            self.fig = go.FigureWidget(data=self.data, layout=self.layout)
            for trace in self.fig.data[:-2]:
                trace.on_click(self.show_data_3d_callback)

    def setup_widgets(self, image_folder='data/images'):
        image_folder = self.base_path + '/' + image_folder
        # self.image_list = os.listdir(image_folder)
        self.image_list = json.load(open('./data/image_list.json', 'r'))
        self.image_list = self.image_list['image_list']
        # Data Filter Setup
        reset_plot = Button(
            description='Reset',
            disabled=False,
            tooltip='Reset the colors of the plot'
        )

        xy_check = Button(
            description='Show X-Y axes',
            disabled=False,
            button_style='',
            tooltip='Click to show X-Y axes'
        )

        show_success_hull = ToggleButton(
            value=True,
            description='Show success hull',
            disabled=False,
            button_style='',
            tooltip='Toggle to show/hide success hull',
            icon='check'
        )

        show_hull_check = ToggleButton(
            value=True,
            description='Show State Space',
            disabled=False,
            button_style='',
            tooltip='Toggle to show/hide state space',
            icon='check'
        )
        unique_inchis = self.full_perovskite_data['_rxn_organic-inchikey'].unique(
        )

        self.select_amine = Dropdown(
            options=[row['Chemical Name'] for
                     i, row in self.inchis.iterrows()
                     if row['InChI Key (ID)'] in unique_inchis],
            description='Amine:',
            disabled=False,
        )

        reset_plot.on_click(self.reset_plot_callback)
        xy_check.on_click(self.set_xy_camera)
        show_success_hull.observe(self.toggle_success_mesh, 'value')
        show_hull_check.observe(self.toggle_mesh, 'value')
        self.select_amine.observe(self.select_amine_callback, 'value')

        # Experiment data tab setup
        self.experiment_table = HTML()
        self.experiment_table.value = "Please click on a point"
        "to explore experiment details"

        # self.image_data = {}

        with open("{}/{}".format(image_folder, 'not_found.png'), "rb") as f:
            b = f.read()
            image_data = b

        self.image_widget = Image(
            value=image_data,
            layout=Layout(height='400px', width='650px')
        )

        experiment_view_vbox = VBox(
            [HBox([self.experiment_table, self.image_widget])])

        plot_tabs = Tab([VBox([self.fig,
                               HBox([self.select_amine]),
                               HBox([xy_check, show_success_hull,
                                     show_hull_check,
                                     reset_plot])]),
                         ])
        plot_tabs.set_title(0, 'Chemical Space')

        self.full_widget = VBox([plot_tabs, experiment_view_vbox])
        self.full_widget.layout.align_items = 'center'

    def select_amine_callback(self, state):
        new_amine_name = state['new']
        new_amine_inchi = self.inchi_dict[new_amine_name]
        amine_data = self.full_perovskite_data[
            self.full_perovskite_data['_rxn_organic-inchikey'] ==
            new_amine_inchi]
        self.current_amine_inchi = new_amine_inchi
        self.generate_plot(self.current_amine_inchi)
        self.reset_plot_callback(None)

    def get_plate_options(self):
        plates = set()
        for df in self.amine_crystal_dfs:
            for i, row in df.iterrows():
                name = str(row['name'])
                plate_name = '_'.join(name.split('_')[: -1])
                plates.add(plate_name)
        plate_options = []
        for i, plate in enumerate(plates):
            plate_options.append(plate)
        return plate_options

    def generate_table(self, row, columns, column_names):
        table_html = """ <table border="1" style="width:100%;">
                        <tbody>"""
        for i, column in enumerate(columns):
            if isinstance(row[column], str):
                value = row[column].split('_')[-1]
            else:
                value = np.round(row[column], decimals=3)
            table_html += """
                            <tr>
                                <td style="padding: 8px;">{}</td>
                                <td style="padding: 8px;">{}</td>
                            </tr>
                          """.format(column_names[i], value)
        table_html += """
                        </tbody>
                        </table>
                        """
        return table_html

    def toggle_mesh(self, state):
        with self.fig.batch_update():
            self.fig.data[-1].visible = state.new

    def toggle_success_mesh(self, state):
        with self.fig.batch_update():
            self.fig.data[-2].visible = state.new

    def set_xy_camera(self, state):
        camera = dict(
            up=dict(x=0, y=0, z=1),
            center=dict(x=0, y=0, z=0),
            eye=dict(x=0.0, y=0.0, z=2.5)
        )

        self.fig['layout'].update(
            scene=dict(camera=camera),
        )

    def reset_plot_callback(self, b):
        with self.fig.batch_update():
            for i in range(len(self.fig.data[:4])):
                self.fig.data[i].marker.color = self.trace_colors[i]
                self.fig.data[i].marker.size = 4

    def show_data_3d_callback(self, trace, point, selector):
        if point.point_inds and point.trace_index < 4:

            selected_experiment = self.amine_crystal_dfs[
                point.trace_index].iloc[point.point_inds[0]]
            with self.fig.batch_update():
                for i in range(len(self.fig.data[: 4])):
                    color = self.trace_colors[i].split(',')
                    color[-1] = '0.5)'
                    color = ','.join(color)
                    if i == point.trace_index:
                        marker_colors = [color for x in range(len(trace['x']))]
                        marker_colors[point.point_inds[0]
                                      ] = self.trace_colors[i]
                        self.fig.data[i].marker.color = marker_colors
                        self.fig.data[i].marker.size = 6
                    else:
                        self.fig.data[i].marker.color = color
                        self.fig.data[i].marker.size = 4
            self.populate_data(selected_experiment)

    def populate_data(self, selected_experiment):
        name = selected_experiment['RunID_vial']
        img_filename = name+'_side.jpg'
        image_folder = os.path.join('data', 'images')
        if img_filename in self.image_list:
            with open(os.path.join(image_folder, img_filename), "rb") as f:
                b = f.read()
                # self.image_data[img_filename] = b
                self.image_widget.value = b
        else:
            with open(os.path.join(image_folder, 'not_found.png'), "rb") as f:
                b = f.read()
                self.image_widget.value = b
            # self.image_widget.value = self.image_data['not_found.png']
        columns = ['RunID_vial', '_rxn_M_acid', '_rxn_M_inorganic', '_rxn_M_organic',
                   '_rxn_mixingtime1S', '_rxn_mixingtime2S',
                   '_rxn_reactiontimeS', '_rxn_stirrateRPM',
                   '_rxn_temperatureC_actual_bulk']
        column_names = ['Well ID', 'Formic Acid [FAH] (M)', 'Lead Iodide [PbI2] (M)',
                        # 'Dimethylammonium Iodide [Me2NH2I]',
                        '{} (M)'.format(
                            self.chem_dict[self.current_amine_inchi]),
                        'Mixing Time Stage 1 (s)', 'Mixing Time Stage 2 (s)',
                        'Reaction Time (s)', 'Stir Rate (RPM)',
                        'Temperature (C)']

        prefix = '_'.join(name.split('_')[:-1])
        self.selected_plate = prefix
        self.experiment_table.value = '<p>Plate ID:<br> {}</p>'.format(
            prefix) + self.generate_table(selected_experiment.loc[columns],
                                          columns, column_names)

    @property
    def plot(self):
        return self.full_widget
