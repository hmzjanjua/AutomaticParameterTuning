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

    # Initialize parameters
    plotly.tools.set_credentials_file(username='mohsinbokhari_2', api_key='8NIac4jBXLtLrdcCV8OY')
    try:
        write_csv = int(sys.argv[3])
    except ValueError:
        print("Writing to CSV")
        write_csv = True
    except IndexError:
        print("Writing to CSV")
        write_csv = True

    # Read data file
    df1 = pd.read_csv(filecompletepath1, sep=',')
    df2 = pd.read_csv(filecompletepath2, sep=',')

    # Construct analytics
    sales_analytics = pd.DataFrame(df1['Actual'])
    sales_analytics['Predict'] = df2['Predict']
    sales_analytics['Difference'] = (sales_analytics['Predict'] - sales_analytics['Actual']) / \
                                    (sales_analytics['Actual'] + 1)
    sales_analytics['Difference'] = sales_analytics['Difference'].apply(
        lambda i: float('inf') if abs(i) == float('inf') else round(100 * i)
    )
    sales_analytics['ABSDifference'] = sales_analytics['Difference'].apply(lambda i: abs(i))
    sales_analytics['Error'] = sales_analytics['ABSDifference'] >= 25
    sales_analytics['month'] = range(1, len(sales_analytics) + 1)
    sales_analytics = sales_analytics.reset_index(drop=True)

    # Write CSV
    if write_csv:
        sales_analytics_out = sales_analytics[['month', 'Actual', 'Predict']]
        sales_analytics_out.to_csv(os.path.join(filepath, filename.split('.')[0]+'.csv'), sep=',', index=False)

    # Plot predicted and actual
    plt.plot(df1['month'], df1['Actual'].apply(lambda i: i * 1.25), 'k', label='Upper Bound', linestyle=':')
    plt.plot(df1['month'], df1['Actual'].apply(lambda i: i * 0.75), 'k', label='Lower Bound', linestyle=':')
    plt.plot(df1['month'], df1['Actual'], 'r', label='Actual', linestyle='-')
    plt.plot(df2['month'], df2['Predict'], 'b', label='Predict', linestyle='-')
    plt.ylabel('Sale')
    plt.xlabel('Month')
    plt.title('Sale-Month')
    plt.gcf().canvas.set_window_title('Sale-Month')
    plt.gcf().set_size_inches(12, 7)
    plt.legend(loc='upper left')
    plt.show()


if __name__ == '__main__':
    main()
