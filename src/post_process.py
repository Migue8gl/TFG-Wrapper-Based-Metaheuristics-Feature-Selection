import matplotlib.pyplot as plt
import pandas as pd
from constants import (
    IMG_DIR,
    KNN_CLASSIFIER,
    OPTIMIZER_COLOR,
    RESULTS_DIR,
    SVC_CLASSIFIER,
)
from plots import plot_grouped_boxplots, plot_rankings


def plot_all_optimizers(df_analysis_b: pd.DataFrame,
                        df_analysis_r: pd.DataFrame,
                        optimizer_color: dict = OPTIMIZER_COLOR):
    """
    Plots boxplots for all optimizers for a given dataset and classifier.

    Parameters:
        - df_analysis_b (pandas.DataFrame): Dataframe with binary analysis results.
        - df_analysis_r (pandas.DataFrame): Dataframe with real analysis results.
        - classifiers (list): List of classifiers to plot.
        - optimizer_color (dict): Dictionary with optimizer:color pairs.
    """

    def _plot_optimizers(df_encoding: pd.DataFrame, encoding: str,
                         dataset_name: str, classifier: str):
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


def make_rankings_for_optimizers(df_analysis_b: pd.DataFrame,
                                 df_analysis_r: pd.DataFrame):
    """
    Makes rankings for all optimizers in all datasets. Generates images with rankings and csv containing rankings.

    Parameters:
        - df_analysis_b (pandas.DataFrame): Dataframe with binary analysis results.
        - df_analysis_r (pandas.DataFrame): Dataframe with real analysis results.
    """
    real_df = df_analysis_r.copy()
    binary_df = df_analysis_b.copy()
    classifiers = ['knn', 'svc']

    for clf in classifiers:
        binary_df_clf = binary_df[binary_df['classifier'] == clf].copy()
        real_df_clf = real_df[real_df['classifier'] == clf].copy()

        # Rank based on avg column
        binary_df_clf['rank_avg'] = binary_df_clf.groupby(
            'dataset')['avg'].rank(method='average')
        real_df_clf['rank_avg'] = real_df_clf.groupby('dataset')['avg'].rank(
            method='average')

        # Rank based on selected_rate column
        binary_df_clf['rank_selected_rate'] = binary_df_clf.groupby(
            'dataset')['selected_rate'].rank(method='average')
        real_df_clf['rank_selected_rate'] = real_df_clf.groupby(
            'dataset')['selected_rate'].rank(method='average')

        # Pivot tables for avg
        pivot_binary_avg = binary_df_clf.pivot_table(index='dataset',
                                                     columns='optimizer',
                                                     values='rank_avg')
        pivot_real_avg = real_df_clf.pivot_table(index='dataset',
                                                 columns='optimizer',
                                                 values='rank_avg')
        # Pivot tables for selected_rate
        pivot_binary_selected_rate = binary_df_clf.pivot_table(
            index='dataset', columns='optimizer', values='rank_selected_rate')
        pivot_real_selected_rate = real_df_clf.pivot_table(
            index='dataset', columns='optimizer', values='rank_selected_rate')

        # Calculate column mean for avg
        binary_mean_avg = pivot_binary_avg.mean().round(2)
        real_mean_avg = pivot_real_avg.mean().round(2)

        # Calculate column mean for selected_rate
        binary_mean_selected_rate = pivot_binary_selected_rate.mean().round(2)
        real_mean_selected_rate = pivot_real_selected_rate.mean().round(2)

        # Add mean row to the pivot tables for avg
        pivot_binary_avg.loc['Mean'] = binary_mean_avg
        pivot_real_avg.loc['Mean'] = real_mean_avg

        # Add mean row to the pivot tables for selected_rate
        pivot_binary_selected_rate.loc['Mean'] = binary_mean_selected_rate
        pivot_real_selected_rate.loc['Mean'] = real_mean_selected_rate

        # Save to CSV for avg
        pivot_binary_avg.to_csv(f'{RESULTS_DIR}binary/rankings_{clf}_avg.csv')
        pivot_real_avg.to_csv(f'{RESULTS_DIR}real/rankings_{clf}_avg.csv')

        # Save to CSV for selected_rate
        pivot_binary_selected_rate.to_csv(
            f'{RESULTS_DIR}binary/rankings_{clf}_selected_rate.csv')
        pivot_real_selected_rate.to_csv(
            f'{RESULTS_DIR}real/rankings_{clf}_selected_rate.csv')


def main():
    df_analysis_b = pd.read_csv(RESULTS_DIR + 'binary/analysis_results.csv')
    df_analysis_r = pd.read_csv(RESULTS_DIR + 'real/analysis_results.csv')

    make_rankings_for_optimizers(df_analysis_b, df_analysis_r)

    real_ranking_svc_avg = pd.read_csv(RESULTS_DIR +
                                       'real/rankings_svc_avg.csv')
    binary_ranking_svc_avg = pd.read_csv(RESULTS_DIR +
                                         'binary/rankings_svc_avg.csv')

    real_ranking_knn_avg = pd.read_csv(RESULTS_DIR +
                                       'real/rankings_knn_avg.csv')
    binary_ranking_knn_avg = pd.read_csv(RESULTS_DIR +
                                         'binary/rankings_knn_avg.csv')

    real_ranking_svc_selected_rate = pd.read_csv(
        RESULTS_DIR + 'real/rankings_svc_selected_rate.csv')
    binary_ranking_svc_selected_rate = pd.read_csv(
        RESULTS_DIR + 'binary/rankings_svc_selected_rate.csv')

    real_ranking_knn_selected_rate = pd.read_csv(
        RESULTS_DIR + 'real/rankings_knn_selected_rate.csv')
    binary_ranking_knn_selected_rate = pd.read_csv(
        RESULTS_DIR + 'binary/rankings_knn_selected_rate.csv')

    # Plot rankings for avg
    plot_rankings(real_ranking_svc_avg, 'Real ranking - svc (avg)')
    plt.savefig(IMG_DIR + 'real/real_rankings_svc_avg.png')

    plot_rankings(binary_ranking_svc_avg, 'Binary ranking - svc (avg)')
    plt.savefig(IMG_DIR + 'binary/binary_rankings_svc_avg.png')

    plot_rankings(real_ranking_knn_avg, 'Real ranking - knn (avg)')
    plt.savefig(IMG_DIR + 'real/real_rankings_knn_avg.png')

    plot_rankings(binary_ranking_knn_avg, 'Binary ranking - knn (avg)')
    plt.savefig(IMG_DIR + 'binary/binary_rankings_knn_avg.png')

    # Plot rankings for selected_rate
    plot_rankings(real_ranking_svc_selected_rate,
                  'Real ranking - svc (selected_rate)')
    plt.savefig(IMG_DIR + 'real/real_rankings_svc_selected_rate.png')

    plot_rankings(binary_ranking_svc_selected_rate,
                  'Binary ranking - svc (selected_rate)')
    plt.savefig(IMG_DIR + 'binary/binary_rankings_svc_selected_rate.png')

    plot_rankings(real_ranking_knn_selected_rate,
                  'Real ranking - knn (selected_rate)')
    plt.savefig(IMG_DIR + 'real/real_rankings_knn_selected_rate.png')

    plot_rankings(binary_ranking_knn_selected_rate,
                  'Binary ranking - knn (selected_rate)')
    plt.savefig(IMG_DIR + 'binary/binary_rankings_knn_selected_rate.png')


if __name__ == '__main__':
    main()
