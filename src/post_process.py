import matplotlib.pyplot as plt
import pandas as pd
from constants import RESULTS_DIR, KNN_CLASSIFIER, SVC_CLASSIFIER
from plots import plot_grouped_boxplots


def main():
    # Read all results
    df_analysis_b = pd.read_csv(RESULTS_DIR + 'binary/analysis_results.csv')
    df_analysis_r = pd.read_csv(RESULTS_DIR + 'real/analysis_results.csv')

    classifiers = [SVC_CLASSIFIER, KNN_CLASSIFIER]

    for encoding in ['binary', 'real']:
        df_encoding = df_analysis_b if encoding == 'binary' else df_analysis_r
        for dataset_name in df_encoding['dataset'].unique():
            for classifier in classifiers:
                filter_dict = {
                    'dataset': dataset_name,
                    'classifier': classifier
                }

                fig_fitness = plot_grouped_boxplots(
                    df_encoding,
                    x='optimizer',
                    filter=filter_dict,
                    title=
                    f'Boxplot Grouped by Optimizer - {encoding} - {classifier} - {dataset_name}',
                    ylabel='Average Fitness')
                plt.savefig(
                    f'{RESULTS_DIR}{encoding}/{dataset_name}/optimizer_boxplot_fitness_{classifier}_{encoding[0]}.png'
                )
                plt.close(fig_fitness)


if __name__ == '__main__':
    main()
