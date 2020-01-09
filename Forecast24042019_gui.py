import random
import os
from math import exp

import matplotlib
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
from matplotlib import style
# import matplotlib.pyplot as plt

import pandas as pd

import tkinter as tk
from tkinter import ttk
from tkinter import filedialog

matplotlib.use("TkAgg")
LARGE_FONT = ("Verdana", 24)
NORMAL_FONT = ("Verdana", 12)
TRANSPARENT_FONT = ("Verdana", 10)
style.use("ggplot")


class SalesForecaster(tk.Tk):

    def __init__(self, *args, **kwargs):
        tk.Tk.__init__(self, *args, **kwargs)

        tk.Tk.iconbitmap(self, default="salesforecasticon.ico")
        tk.Tk.wm_title(self, "Sales Forecaster")

        self.attributes("-fullscreen", True)
        self.resizable(False, False)

        container = tk.Frame(self)
        container.pack(side="top", fill="both", expand=True)
        container.grid_rowconfigure(0, weight=1)
        container.grid_columnconfigure(0, weight=1)

        frame = ForecastPage(container, self)
        frame.grid(row=0, column=0, sticky="nsew")
        frame.configure(background='white')

        self.__class__.show_frame(frame)

    @staticmethod
    def show_frame(frame):
        frame.tkraise()


class ForecastPage(tk.Frame):

    def __init__(self, parent, controller):
        tk.Frame.__init__(self, parent)

        # Initialize parameters
        self.var_m_d = tk.DoubleVar()
        self.var_c_d = tk.DoubleVar()
        self.var_k = tk.DoubleVar()
        self.var_a = tk.DoubleVar()
        self.var_m_h = tk.DoubleVar()
        self.var_c_h = tk.DoubleVar()
        self.pars = [self.var_m_d, self.var_c_d, self.var_k, self.var_a, self.var_m_h, self.var_c_h]

        self.default_path = './'
        self.filepath = None
        self.filename = None
        self.filecompletepath = None
        self.configfilecompletepath = None
        self.config_pars = None
        self.configfile = None

        self.df = None
        self.sales_analytics_out = None

        # Setup grid for positioning
        self.grid_rowconfigure(0, weight=1)
        self.grid_columnconfigure(0, weight=1)

        # Create fancy header
        sales_forecaster_label = tk.Label(self, text="Sales Forecaster", font=LARGE_FONT)
        sales_forecaster_label.grid(row=0, column=0, columnspan=4)
        sales_forecaster_label.configure(background='white')
        self.header_label = sales_forecaster_label

        # Create quit button
        quit_button = ttk.Button(self, text="Quit (q)", command=lambda: quit())
        quit_button.grid(row=2, column=3, sticky='SE', ipadx=10, ipady=10)

        # Create matplotlib graph
        self.fig = Figure(dpi=100, figsize=(14, 8), tight_layout=True)
        self.plt = self.fig.add_subplot(111)
        self.plt.plot([1, 2, 3, 4, 5, 6, 7, 8, 9], [4, 6, 9, 7, 1, 5, 2, 3, 8])

        self.canvas = FigureCanvasTkAgg(self.fig, self)
        self.canvas.get_tk_widget().grid(row=1, column=0, rowspan=2, ipadx=10, ipady=10)
        self.canvas.draw()

        # Create file open buttons
        config_file_open_button = ttk.Button(self, text="Open Config File",
                                             command=lambda: self.open_config_file_dialog())
        config_file_open_button.grid(row=2, column=1, columnspan=2, sticky='NW', ipadx=10, ipady=10)

        data_file_open_button = ttk.Button(self, text="Open Data File", command=lambda: self.open_data_file_dialog())
        data_file_open_button.grid(row=2, column=1, columnspan=2, sticky='W', ipadx=10, ipady=10)

        # Create save buttons
        config_save_button = ttk.Button(self, text="Save Config File", command=lambda: self.save_config())
        config_save_button.grid(row=2, column=2, columnspan=2, sticky='NE', ipadx=10, ipady=10)

        data_save_button = ttk.Button(self, text="Save Data Files", command=lambda: self.save_data())
        data_save_button.grid(row=2, column=2, columnspan=2, sticky='E', ipadx=10, ipady=10)

        # Create parameter tuning GUI objects
        parameters_label = tk.Label(self, text="Parameters", font=NORMAL_FONT)
        parameters_label.grid(row=0, column=1, columnspan=2, sticky='S')
        parameters_label.configure(background='white')

        space_label = tk.Label(self, text="<space>", font=TRANSPARENT_FONT)
        space_label.grid(row=1, column=2)
        space_label.configure(background='white', foreground='white')

        self.m_d_scale = tk.Scale(self, variable=self.var_m_d, orient='vertical', from_=100, to=-100, label="m_d",
                                  background='white')
        self.m_d_scale.grid(row=1, column=1, sticky='NW')
        self.m_d_scale.bind('<ButtonRelease-1>', self.forecast)

        self.c_d_scale = tk.Scale(self, variable=self.var_c_d, orient='vertical', from_=100, to=-100, label="c_d",
                                  background='white')
        self.c_d_scale.grid(row=1, column=3, sticky='NE')
        self.c_d_scale.bind('<ButtonRelease-1>', self.forecast)

        self.k_scale = tk.Scale(self, variable=self.var_k, orient='vertical', from_=100, to=-100, label="k",
                                background='white')
        self.k_scale.grid(row=1, column=1, sticky='W')
        self.k_scale.bind('<ButtonRelease-1>', self.forecast)

        self.a_scale = tk.Scale(self, variable=self.var_a, orient='vertical', from_=100, to=-100, label="a",
                                background='white')
        self.a_scale.grid(row=1, column=3, sticky='E')
        self.a_scale.bind('<ButtonRelease-1>', self.forecast)

        self.m_h_scale = tk.Scale(self, variable=self.var_m_h, orient='vertical', from_=100, to=-100, label="m_h",
                                  background='white')
        self.m_h_scale.grid(row=1, column=1, sticky='SW')
        self.m_h_scale.bind('<ButtonRelease-1>', self.forecast)

        self.c_h_scale = tk.Scale(self, variable=self.var_c_h, orient='vertical', from_=100, to=-100, label="c_h",
                                  background='white')
        self.c_h_scale.grid(row=1, column=3, sticky='SE')
        self.c_h_scale.bind('<ButtonRelease-1>', self.forecast)

        self.pars_gadgets = [self.m_d_scale, self.c_d_scale, self.k_scale, self.a_scale, self.m_h_scale, self.c_h_scale]

        # Open data file and config file dialogs
        self.open_data_file_dialog()
        self.open_config_file_dialog()

        # Bind 'q' key to quit() function
        controller.bind('q', lambda event: quit())

    def forecast(self, event=None):
        assert event is not None

        if (self.filecompletepath is None) or (self.configfilecompletepath is None) or (self.df is None):
            return

        self.update_scales()

        # Read parameters from config file
        grad = self.var_m_d.get()
        inter = self.var_c_d.get()
        hfact = self.var_m_h.get()
        hconst = self.var_c_h.get()
        k = self.var_k.get()
        a = self.var_a.get()

        filename = os.path.basename(self.filecompletepath)
        self.filename = filename
        self.filepath = os.path.dirname(self.filecompletepath)

        # Read data file
        df = self.df

        # Add Date related columns
        df['month'] = [x[5:7] for x in df['Date']]
        df['year'] = [x[0:4] for x in df['Date']]

        # Train Test Split
        # df_train = df[(df['year'] == '2016') | (df['year'] == '2017')]
        df_test = df[(df['year'] == '2018') | (df['year'] == '2019')]
        # df_test = df

        # Construct a model
        preds = pd.DataFrame()
        preds['linear'] = df_test['DiscountPerc'].apply(lambda i: i * grad).apply(lambda j: j + inter)
        preds['change'] = df_test['Campaign'].apply(lambda i: k * (1 - exp(-a * i)))
        preds['change'] = (preds['change'] * preds['linear']) + df_test['Holiday'].apply(lambda i: i * hfact) + hconst
        preds['pred'] = preds['linear'] + preds['change']

        # Construct analytics
        sales_analytics = pd.DataFrame(df_test['Date'])
        sales_analytics['Actual'] = df_test['Sale']
        sales_analytics['Predict'] = preds['pred']
        sales_analytics['Difference'] = (sales_analytics['Predict'] - sales_analytics['Actual']) / \
                                        (sales_analytics['Actual'] + 1)
        sales_analytics['Difference'] = sales_analytics['Difference'].apply(
            lambda i: float('inf') if abs(i) == float('inf') else round(100 * i)
        )
        sales_analytics['ABSDifference'] = sales_analytics['Difference'].apply(lambda i: abs(i))
        sales_analytics['Error'] = sales_analytics['ABSDifference'] >= 25
        sales_analytics['month'] = range(1, len(sales_analytics) + 1)
        sales_analytics['Holiday'] = df_test['Holiday']
        sales_analytics['Campaign'] = df_test['Campaign']
        sales_analytics['DiscountPerc'] = df_test['DiscountPerc']
        sales_analytics = sales_analytics.reset_index(drop=True)

        self.sales_analytics_out = sales_analytics[['month', 'Actual', 'Predict']]

        error_analytics = self.detect_outliers(sales_analytics)
        self.plot_graph(sales_analytics, error_analytics)

    def plot_graph(self, sales_analytics, error_analytics):
        # Plot predicted and actual
        self.plt.clear()
        self.plt.plot(sales_analytics['month'], sales_analytics['Actual'].apply(lambda i: i * 1.25), 'k',
                      label='Upper Bound',
                      linestyle=':')
        self.plt.plot(sales_analytics['month'], sales_analytics['Actual'].apply(lambda i: i * 0.75), 'k',
                      label='Lower Bound',
                      linestyle=':')
        self.plt.plot(sales_analytics['month'], sales_analytics['Actual'], 'r', label='Actual', linestyle='-')
        self.plt.plot(sales_analytics['month'], sales_analytics['Predict'], 'b', label='Predict')
        self.plt.plot(error_analytics['month'], error_analytics['Predict'], 'g', label='Out Of Bounds',
                      linestyle="None", marker='o')
        self.plt.set_ylabel('Sale')
        self.plt.set_xlabel('Month')
        self.plt.set_title('Sale-Month for ' + str(self.filename.split('.')[0]))
        self.plt.legend(loc='upper left')

        self.canvas.draw()

    def detect_outliers(self, sales_analytics):
        assert isinstance(self, ForecastPage)
        error_analytics = sales_analytics[
            (abs(sales_analytics['Actual']-sales_analytics['Predict'])/sales_analytics['Actual']) >= 0.25
        ]
        return error_analytics

    def update_scales(self):
        for i, gadget in enumerate(self.pars_gadgets):
            par_value = self.pars[i].get()
            if abs(par_value) > 0:
                if par_value > 0:
                    gadget.configure(from_=(3*par_value), to=-par_value, resolution=float(round(abs(par_value)/1000)))
                else:
                    gadget.configure(from_=-par_value, to=(3*par_value), resolution=float(round(abs(par_value)/1000)))
            else:
                gadget.configure(from_=100, to=-100)

    def open_config_file_dialog(self):
        self.configfilecompletepath = filedialog.askopenfilename(initialdir=self.default_path,
                                                                 title="Select configuration file",
                                                                 filetypes=(("CSV files", "*.csv"),))

        try:
            configfile = pd.read_csv(self.configfilecompletepath, sep=',')
        except FileNotFoundError:
            configfile = None
            print("Config File Does Not Exist")
        self.configfile = configfile

        if (self.filename is not None) and (self.configfile is not None):
            config = configfile[configfile['filename'] == self.filename]
            assert (len(config) == 1)

            self.load_config(config)
            self.forecast(event=1)

    def load_config(self, config):
        self.config_pars = [float(config['gradient'].astype(float)), float(config['intercept'].astype(float)),
                            float(config['campaignfactor'].astype(float)),
                            float(config['campaignexponentfactor'].astype(float)),
                            float(config['holidayfactor'].astype(float)),
                            float(config['holidayconstant'].astype(float))]
        for i, gadget in enumerate(self.pars_gadgets):
            par_value = self.config_pars[i]
            if abs(par_value) > 0:
                if par_value > 0:
                    gadget.configure(from_=(3*par_value), to=-par_value, resolution=float(round(abs(par_value)/1000)))
                else:
                    gadget.configure(from_=-par_value, to=(3*par_value), resolution=float(round(abs(par_value)/1000)))
            else:
                gadget.configure(from_=100, to=-100)
            self.pars[i].set(self.config_pars[i])

    def open_data_file_dialog(self):
        self.filecompletepath = filedialog.askopenfilename(initialdir=self.default_path,
                                                           title="Select data file",
                                                           filetypes=(("text files", "*.txt"),
                                                                      ("tab separated files", "*.tsv"),))
        self.filename = os.path.basename(self.filecompletepath)
        self.filepath = os.path.dirname(self.filecompletepath)

        # Read data file
        try:
            self.df = pd.read_csv(self.filecompletepath, sep='\t')
        except FileNotFoundError:
            self.df = None
            self.filename = None
            self.filepath = None
            self.filecompletepath = None
            return

        if self.configfile is not None:
            config = self.configfile[self.configfile['filename'] == self.filename]
            assert (len(config) == 1)

            self.load_config(config)
            self.forecast(event=1)

    def save_config(self):
        if self.configfile is None:
            return
        self.configfile.loc[self.configfile.filename == self.filename, 'gradient'] = self.var_m_d.get()
        self.configfile.loc[self.configfile.filename == self.filename, 'intercept'] = self.var_c_d.get()
        self.configfile.loc[self.configfile.filename == self.filename, 'campaignfactor'] = self.var_k.get()
        self.configfile.loc[self.configfile.filename == self.filename, 'campaignexponentfactor'] = self.var_a.get()
        self.configfile.loc[self.configfile.filename == self.filename, 'holidayfactor'] = self.var_m_h.get()
        self.configfile.loc[self.configfile.filename == self.filename, 'holidayconstant'] = self.var_c_h.get()

        self.configfile.to_csv(self.configfilecompletepath, sep=',', index=False)

    def save_data(self):
        if (self.filename is None) or (self.filepath is None):
            return
        self.fig.savefig(os.path.join(self.filepath, self.filename.split('.')[0] + '.png'))

        if self.sales_analytics_out is None:
            return
        self.sales_analytics_out.to_csv(
            os.path.join(self.filepath, self.filename.split('.')[0] + '.csv'),
            sep=',',
            index=False
        )

    def random_plot(self):
        rand_data_x = [1, 2, 3, 4, 5, 6, 7, 8, 9]
        rand_data_y = rand_data_x.copy()
        random.shuffle(rand_data_y)
        self.plt.clear()
        self.plt.plot(rand_data_x, rand_data_y)
        self.canvas.draw()

    @staticmethod
    def getpercentrange(num, x):
        assert type(x) == float
        assert type(num) == float
        assert 0 <= x <= 100

        x_percent = abs((x * num) / 100)
        output = [x_percent, num - x_percent, num + x_percent]
        return output


app = SalesForecaster()
app.mainloop()
