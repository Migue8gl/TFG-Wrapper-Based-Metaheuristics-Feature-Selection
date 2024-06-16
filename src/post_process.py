import json
import os

import matplotlib.pyplot as plt
import pandas as pd

from constants import IMG_DIR, RESULTS_DIR
from plots import (plot_all_boxplots_optimizers, plot_fitness_all_optimizers,
                   plot_rankings, plot_mean_metrics_comparison)


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
    """for encoding in ['binary', 'real']:
        for dataset_name in df_analysis_b['dataset'].unique():
            for classifier in df_analysis_b['classifier'].unique():
                with open(
                        os.path.join(RESULTS_DIR, encoding, dataset_name,
                                     f'all_fitness_{classifier}.json'),
                        'r') as file:
                    fitness_register = json.load(file)
                plot_fitness_all_optimizers(fitness_register)
                plt.savefig(
                    os.path.join(IMG_DIR, encoding, dataset_name,
                                 f'optimizers_fitness_{classifier}.png'))
                plt.close()

    # Create directory to store dataset metrics images
    img_directory_path = os.path.join(IMG_DIR, 'real')
    if not os.path.isdir(img_directory_path):
        os.makedirs(img_directory_path)
    img_directory_path = os.path.join(IMG_DIR, 'binary')
    if not os.path.isdir(img_directory_path):
        os.makedirs(img_directory_path)

    # Create directory to store dataset metrics results
    result_path = os.path.join(RESULTS_DIR, 'real')
    if not os.path.isdir(result_path):
        os.makedirs(result_path)
    result_path = os.path.join(RESULTS_DIR, 'binary')
    if not os.path.isdir(result_path):
        os.makedirs(result_path)

    plot_all_boxplots_optimizers(df_analysis_b, df_analysis_r)

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
    plt.savefig(IMG_DIR + 'real/rankings_svc_avg.png')

    plot_rankings(binary_ranking_svc_avg, 'Binary ranking - svc (avg)')
    plt.savefig(IMG_DIR + 'binary/rankings_svc_avg.png')

    plot_rankings(real_ranking_knn_avg, 'Real ranking - knn (avg)')
    plt.savefig(IMG_DIR + 'real/rankings_knn_avg.png')

    plot_rankings(binary_ranking_knn_avg, 'Binary ranking - knn (avg)')
    plt.savefig(IMG_DIR + 'binary/rankings_knn_avg.png')

    # Plot rankings for selected_rate
    plot_rankings(real_ranking_svc_selected_rate,
                  'Real ranking - svc (selected_rate)')
    plt.savefig(IMG_DIR + 'real/rankings_svc_selected_rate.png')

    plot_rankings(binary_ranking_svc_selected_rate,
                  'Binary ranking - svc (selected_rate)')
    plt.savefig(IMG_DIR + 'binary/rankings_svc_selected_rate.png')

    plot_rankings(real_ranking_knn_selected_rate,
                  'Real ranking - knn (selected_rate)')
    plt.savefig(IMG_DIR + 'real/rankings_knn_selected_rate.png')

    plot_rankings(binary_ranking_knn_selected_rate,
                  'Binary ranking - knn (selected_rate)')
    plt.savefig(IMG_DIR + 'binary/rankings_knn_selected_rate.png')"""

    for metric in ['acc', 'avg', 'selected_rate']:
        plot_mean_metrics_comparison(df_analysis_b, df_analysis_r, metric)
        plt.savefig(os.path.join(IMG_DIR, f'{metric}_comparison.png'))
        plt.close()


if __name__ == '__main__':
    main()
