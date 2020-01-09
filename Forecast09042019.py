import os
import sys
import pandas as pd
import plotly
import matplotlib.pyplot as plt

from math import exp


def main():
    # Set input parameters
    filenames = ['1_BAGS_BASLT.txt', '2_SLG_BASLT.txt', '3_BAGS_NYLON.txt', '4_BAGS_EXOTC.txt', '5_BAGS_NOVEL.txt',
                 '6_BAGS_FASHL.txt', '7_BAGS_ALTO.txt', '8_BAGS_FASHF.txt', '9_BAGS_NFL.txt', '10_BAGS_MLB.txt']
    filepath = 'E:/Dooney/Code/AutomaticParameterTuning/level4/dus'
    filename = ''
    try:
        filenumber = int(sys.argv[1])
        filename = filenames[filenumber - 1]
        filecompletepath = os.path.join(filepath, filename)
    except ValueError:
        filename = sys.argv[1]
        filename = filename.split('\n')[0]
        if os.path.exists(filename):
            filecompletepath = filename
            filename = os.path.basename(filename)
            filepath = os.path.dirname(filecompletepath)
        else:
            if not os.path.exists(os.path.join(filepath, filename)):
                print("File does not exist\nExiting ....")
                exit()
            filecompletepath = os.path.join(filepath, filename)
    except IndexError:
        filecompletepath = ''
        print("Please give correct filenumber or filename")
        exit()

    print(f"Complete File Path:\n{filecompletepath}")

    configfilename = 'config.csv'
    configfilecompletepath = os.path.join(filepath, configfilename)
    write_csv = True
    plotly.tools.set_credentials_file(username='mohsinbokhari_2', api_key='8NIac4jBXLtLrdcCV8OY')

    # Read config file
    print(f"Config File Complete Path:\n{configfilecompletepath}")
    configfile = pd.DataFrame()
    try:
        configfile = pd.read_csv(configfilecompletepath, sep=',')
    except FileExistsError:
        print("Config File Does Not Exist")
        exit()
    config = configfile[configfile['filename'] == filename]
    assert(len(config) == 1)

    # Read parameters from config file
    grad = float(config['gradient'].astype(float))
    inter = float(config['intercept'].astype(float))
    hfact = float(config['holidayfactor'].astype(float))
    hconst = float(config['holidayconstant'].astype(float))
    k = float(config['campaignfactor'].astype(float))
    a = float(config['campaignexponentfactor'].astype(float))

    # Read data file
    df = pd.read_csv(filecompletepath, sep='\t')

    # Add Date related columns
    df['month'] = [x[5:7] for x in df['Month']]
    df['year'] = [x[0:4] for x in df['Month']]

    # Train Test Split
    # df_train = df[(df['year'] == '2016') | (df['year'] == '2017')]
    df_test = df[(df['year'] == '2018') | (df['year'] == '2019')]

    # Construct a model
    preds = pd.DataFrame()
    preds['linear'] = df_test['DiscountPerc'].apply(lambda i: i * grad).apply(lambda j: j + inter)
    preds['change'] = df_test['Campaign'].apply(lambda i: k * (1 - exp(-a * i)))
    preds['change'] = (preds['change'] * preds['linear']) + \
        df_test['Holiday'].apply(lambda i: i * hfact) + hconst  # (exp(i-1)-2)
    preds['pred'] = preds['linear'] + preds['change']

    # Construct analytics
    sales_analytics = pd.DataFrame(df_test['Month'])
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

    # Generate output
    if write_csv:
        sales_analytics_out = sales_analytics[['month', 'Actual', 'Predict']]
        sales_analytics_out.to_csv(os.path.join(filepath, filename.split('.')[0]+'.csv'), sep=',', index=False)

    # Print output
    print(f'grad: {grad}\ninter: {inter}\nhfact: {hfact}\nhconst: {hconst}\nk: {k}\na: {a}')
    print(sales_analytics)

    # Plot predicted and actual
    plt.plot(sales_analytics['month'], sales_analytics['Actual'].apply(lambda i: i * 1.25), 'k', label='Upper Bound',
             linestyle=':')
    plt.plot(sales_analytics['month'], sales_analytics['Actual'].apply(lambda i: i * 0.75), 'k', label='Lower Bound',
             linestyle=':')
    plt.plot(sales_analytics['month'], sales_analytics['Actual'], 'r', label='Actual', linestyle='-')
    plt.plot(sales_analytics['month'], sales_analytics['Predict'], 'b', label='Predict')
    plt.ylabel('Sale')
    plt.xlabel('Month')
    plt.title('Sale-Month for ' + filename.split('.')[0])
    plt.gcf().canvas.set_window_title('Sale-Month for ' + filename.split('.')[0])
    plt.gcf().set_size_inches(12, 7)
    plt.legend(loc='upper left')
    if write_csv:
        plt.savefig(os.path.join(filepath, filename.split('.')[0]+'.png'), bbox_inches='tight')
    plt.show()


if __name__ == '__main__':
    main()
