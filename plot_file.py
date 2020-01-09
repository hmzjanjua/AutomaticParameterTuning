import os
import sys
import pandas as pd
import plotly
import matplotlib.pyplot as plt


def main():
    # Set input parameters
    filenames = ['dus_1_BAGS_BASLT.txt', 'dus_2_SLG_BASLT.txt', 'dus_3_BAGS_NYLON.txt', 'dus_4_BAGS_EXOTC.txt']

    filepath = 'E:/projects/dooney/AutomaticParameterTuning/data/hierarchyLevel4/dus/'
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
            # filepath = os.path.dirname(filecompletepath)
        else:
            if not os.path.exists(os.path.join(filepath, filename)):
                print("File does not exist\nExiting ....")
                exit()
            filecompletepath = os.path.join(filepath, filename)
    except IndexError:
        filecompletepath = ''
        print("Please give correct filenumber or filename")
        exit()

    print(f'Complete File Path:\n{filecompletepath}')

    plotly.tools.set_credentials_file(username='mohsinbokhari_2', api_key='8NIac4jBXLtLrdcCV8OY')

    # Read data file
    df = pd.read_csv(filecompletepath, sep='\t')

    # Plot predicted and actual
    plt.plot(df['Month'], df['Sale'].apply(lambda i: i * 1.25), 'k', label='Upper Bound', linestyle=':')
    plt.plot(df['Month'], df['Sale'].apply(lambda i: i * 0.75), 'k', label='Lower Bound', linestyle=':')
    plt.plot(df['Month'], df['Sale'], 'r', label='Actual', linestyle='-')
    plt.ylabel('Sale')
    plt.xlabel('Month')
    plt.xticks(df['Month'], df['Month'], rotation='vertical')
    plt.subplots_adjust(bottom=0.25)
    plt.title('Sale-Month for ' + filename.split('.')[0])
    plt.gcf().canvas.set_window_title('Sale-Month for ' + filename.split('.')[0])
    plt.gcf().set_size_inches(12, 7)
    plt.legend(loc='upper left')
    plt.show()


if __name__ == '__main__':
    main()
