import pandas as pd
import os
from math import exp
# from azureml.core import Run
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--data-folder', type=str, dest='data_folder', help='data folder mounting point')
args = parser.parse_args()


def perform_sale_forecast(base_dir='.'):

    input_dir = 'input'
    output_dir = 'output'
    test_dir = 'test'
    # train_dir = 'train'
    config_dir = 'config'

    configfilename = 'config.txt'

    inputfilecompletepath = os.path.join(base_dir, input_dir, test_dir)

    input_files = [f for f in os.listdir(inputfilecompletepath)
                   if os.path.isfile(os.path.join(inputfilecompletepath, f))]

    # Read config file
    configfile = pd.read_csv(os.path.join(base_dir, config_dir, configfilename), sep='\t')

    for file_name in input_files:
        # Filter Config
        config = configfile[configfile['filename'] == file_name]

        # Read parameters from config file
        grad = float(config['gradient'].astype(float))
        inter = float(config['intercept'].astype(float))
        hfact = float(config['holidayfactor'].astype(float))
        hconst = float(config['holidayconstant'].astype(float))
        k = float(config['campaignfactor'].astype(float))
        a = float(config['campaignexponentfactor'].astype(float))
        bos = config['usebos'].astype(bool).all()

        df = pd.read_csv(os.path.join(inputfilecompletepath, file_name), sep='\t')

        # Construct a model
        df['linear'] = df['DiscountPerc'].apply(lambda i: i * grad).apply(lambda j: j + inter)
        df['change'] = df['Campaign'].apply(lambda i: k * (1 - exp(-a * i)))
        df['change'] = (df['change'] * df['linear']) + \
            df['Holiday'].apply(lambda i: i * hfact) + hconst
        df['PredictedSale'] = df['linear'] + df['change']
        df['Month'] = df['Month']

        # Apply BOS
        if bos:
            df['PredictedSale'] = df['PredictedSale'].apply(lambda i: i*12)[0] * df['bos']

        df.to_csv(os.path.join(base_dir, output_dir, file_name),
                  columns=['Month', 'PredictedSale'], sep='\t', index=False)


os.makedirs(os.path.join(args.data_folder, 'forsaleforecasting/output'), exist_ok=True)

perform_sale_forecast(os.path.join(args.data_folder, 'forsaleforecasting'))
