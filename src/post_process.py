import matplotlib.pyplot as plt
import pandas as pd
from constants import RESULTS_DIR, KNN_CLASSIFIER, SVC_CLASSIFIER
from plots import plot_grouped_boxplots, plot_rankings


def plot_all_optimizers(df_analysis_b, df_analysis_r, optimizer_color):
    """
    Plots boxplots for all optimizers for a given dataset and classifier.

    Parameters:
        - df_analysis_b (pandas.DataFrame): Dataframe with binary analysis results.
        - df_analysis_r (pandas.DataFrame): Dataframe with real analysis results.
        - classifiers (list): List of classifiers to plot.
        - optimizer_color (dict): Dictionary with optimizer:color pairs.
    """

    def _plot_optimizers(df_encoding, encoding, dataset_name, classifier):
        filter_dict = {'dataset': dataset_name, 'classifier': classifier}

        fig_fitness = plot_grouped_boxplots(
            df_encoding,
            x='optimizer',
            x_color=optimizer_color,
            filter=filter_dict,
            title=
            f'Boxplot Grouped by Optimizer - {encoding} - {classifier} - {dataset_name}',
            ylabel='Average Fitness')
        plt.savefig(
            f'{RESULTS_DIR}{encoding}/{dataset_name}/optimizer_boxplot_fitness_{classifier}_{encoding[0]}.png'
        )
        plt.close(fig_fitness)

    classifiers = [SVC_CLASSIFIER, KNN_CLASSIFIER]
    for encoding in ['binary', 'real']:
        df_encoding = df_analysis_b if encoding == 'binary' else df_analysis_r
        for dataset_name in df_encoding['dataset'].unique():
            for classifier in classifiers:
                _plot_optimizers(df_encoding, encoding, dataset_name,
                                 classifier)


def make_rankings_for_optimizers(df_analysis_b, df_analysis_r):
    real_df = df_analysis_r.copy()
    binary_df = df_analysis_b.copy()
    classifiers = [KNN_CLASSIFIER, SVC_CLASSIFIER]

    for clf in classifiers:
        ranking_r = real_df[real_df['classifier'] == clf].groupby(
            'optimizer')['avg'].mean().sort_values(ascending=True)
        ranking_b = binary_df[binary_df['classifier'] == clf].groupby(
            'optimizer')['avg'].mean().sort_values(ascending=True)

        ranking_r.to_csv(RESULTS_DIR + 'real/rankings_{}.csv'.format(clf),
                         index=True)
        ranking_b.to_csv(RESULTS_DIR + 'binary/rankings_{}.csv'.format(clf),
                         index=True)

    ranking_b = binary_df.groupby('optimizer')['avg'].mean().sort_values(
        ascending=True)
    ranking_r = real_df.groupby('optimizer')['avg'].mean().sort_values(
        ascending=True)
    ranking_b.to_csv(RESULTS_DIR + 'binary/rankings.csv', index=True)
    ranking_r.to_csv(RESULTS_DIR + 'real/rankings.csv', index=True)


def main():
    df_analysis_b = pd.read_csv(RESULTS_DIR + 'binary/analysis_results.csv')
    df_analysis_r = pd.read_csv(RESULTS_DIR + 'real/analysis_results.csv')

    optimizer_color = {
        'gwo': 'darkred',
        'goa': 'darkgreen',
        'fa': 'navy',
        'cs': 'darkorange',
        'ga': 'indigo',
        'woa': 'darkcyan',
        'abco': 'darkmagenta',
        'da': 'olive',
        'aco': 'deeppink',
        'pso': 'limegreen',
        'ba': 'dodgerblue',
        'de': 'saddlebrown'
    }

    #plot_all_optimizers(df_analysis_b, df_analysis_r,optimizer_color)

    make_rankings_for_optimizers(df_analysis_b, df_analysis_r)

    real_ranking = pd.read_csv(RESULTS_DIR + 'real/rankings.csv')
    binary_ranking = pd.read_csv(RESULTS_DIR + 'binary/rankings.csv')

    plot_rankings(real_ranking, 'Real ranking', optimizer_color)
    plt.savefig(RESULTS_DIR + 'real/real_rankings.png')
    plot_rankings(binary_ranking, 'Binary ranking', optimizer_color)
    plt.savefig(RESULTS_DIR + 'binary/binary_rankings.png')


if __name__ == '__main__':
    main()
