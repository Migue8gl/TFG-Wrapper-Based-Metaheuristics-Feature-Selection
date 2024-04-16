import matplotlib.pyplot as plt
import pandas as pd
from constants import RESULTS_DIR
from plots import plot_grouped_boxplots


def main():
    # Read all results
    df_analysis_b = pd.read_csv(RESULTS_DIR + 'binary/analysis_results.csv')
    df_analysis_r = pd.read_csv(RESULTS_DIR + 'real/analysis_results.csv')

    # Generate boxplot for binary and real results
    for encoding in ['binary', 'real']:
        fig = plot_grouped_boxplots(
            df_analysis_b if encoding == 'binary' else df_analysis_r,
            x='optimizer',
            title=f'Boxplot Grouped by Optimizer - {encoding}',
            ylabel='Average Fitness')
        plt.savefig(RESULTS_DIR +
                    f'{encoding}/optimizer_boxplot_fitness_{encoding[0]}.png')
        plt.close(fig)

        for dataset_name in df_analysis_b['dataset'].unique():
            fig = plot_grouped_boxplots(
                df_analysis_b if encoding == 'binary' else df_analysis_r,
                x='optimizer',
                filter={
                    'col': 'dataset',
                    'val': dataset_name
                },
                title='Boxplot Grouped by Optimizer - {} - {}'.format(
                    encoding, dataset_name),
                ylabel='Average Fitness')
            plt.savefig(RESULTS_DIR + f'{encoding}/{dataset_name}/' +
                        'optimizer_boxplot_fitness_{}.png'.format(encoding[0]))
            plt.close(fig)

           


if __name__ == '__main__':
    main()
