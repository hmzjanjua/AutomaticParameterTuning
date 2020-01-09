import os
import sys
import pandas as pd
import plotly
import matplotlib.pyplot as plt


def main():
    # Set input path
    filecompletepath1 = sys.argv[1]
    filecompletepath2 = sys.argv[2]

    filepath = os.path.dirname(filecompletepath1)
    filename = os.path.basename(filecompletepath1).split('.')[0] + os.path.basename(filecompletepath2).split('.')[0]

    if not (os.path.exists(filecompletepath1) and os.path.exists(filecompletepath2)):
        print("Input file path does not exist")
        exit()

    print(f"Complete File Path 1:\n{filecompletepath1}")
    print(f"Complete File Path 2:\n{filecompletepath2}")

    plotly.tools.set_credentials_file(username='mohsinbokhari_2', api_key='8NIac4jBXLtLrdcCV8OY')

    # Read data file
    df1 = pd.read_csv(filecompletepath1, sep=',')
    df2 = pd.read_csv(filecompletepath2, sep=',')

    # Combine data
    df = pd.DataFrame(df1['Predict'] + df2['Predict'])
    df['Actual'] = df1['Actual'] + df2['Actual']
    df['month'] = df1['month']

    # Generate output
    df.to_csv(os.path.join(filepath, f'{filename}.csv'), sep=',', index=False)

    # Plot predicted and actual
    plt.plot(df1['month'], df1['Predict'], 'b', label='File 1')
    plt.plot(df2['month'], df2['Predict'], 'r', label='File 2')
    plt.plot(df['month'], df['Predict'], 'k', label='Output')
    plt.ylabel('Sale')
    plt.xlabel('Month')
    plt.title('Sale-Month')
    plt.gcf().canvas.set_window_title('Sale-Month')
    plt.gcf().set_size_inches(12, 7)
    plt.legend(loc='upper left')
    plt.show()


if __name__ == '__main__':
    main()
