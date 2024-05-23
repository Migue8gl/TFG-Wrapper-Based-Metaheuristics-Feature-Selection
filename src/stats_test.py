import json
import os

import pandas as pd
from constants import RESULTS_DIR
from scipy.stats import wilcoxon


def wilcoxon_test_optimizer(grouped_df_binary, grouped_df_real, column='avg'):
    p_values = {}
    for optimizer, group in grouped_df_binary:
        if optimizer != 'aco':
            avg_opt1 = group[column]
            avg_opt2 = grouped_df_real.get_group(optimizer)[column]

            _, p_value = wilcoxon(avg_opt1, avg_opt2)
            p_values[optimizer] = p_value
    return p_values


def wilcoxon_test_classifier(grouped_df, column='avg'):
    p_values = {}
    knn_group = grouped_df.get_group('knn')
    svc_group = grouped_df.get_group('svc')

    avg_knn = knn_group[column]
    avg_svc = svc_group[column]

    _, p_value = wilcoxon(avg_knn, avg_svc)
    p_values['knn_vs_svc'] = p_value
    return p_values


def main():
    df_binary = pd.read_csv(RESULTS_DIR + 'binary/analysis_results.csv')
    df_real = pd.read_csv(RESULTS_DIR + 'real/analysis_results.csv')

    if not os.path.exists(os.path.join(RESULTS_DIR, 'stats')):
        os.makedirs(os.path.join(RESULTS_DIR, 'stats'))

    for column in ['avg', 'acc', 'selected_rate', 'execution_time']:
        grouped_binary = df_binary.groupby('optimizer')
        grouped_real = df_real.groupby('optimizer')

        p_values = wilcoxon_test_optimizer(grouped_binary, grouped_real,
                                           column)
        with open(
                os.path.join(RESULTS_DIR, 'stats',
                             f'wilcoxon_{column}_optimizer.txt'), 'w') as f:
            json.dump(p_values, f, indent=4)

        grouped_binary = df_binary.groupby('classifier')
        grouped_real = df_real.groupby('classifier')
        p_values = wilcoxon_test_classifier(grouped_binary, column)
        with open(
                os.path.join(RESULTS_DIR, 'stats',
                             f'wilcoxon_{column}_classifier_binary.txt'),
                'w') as f:
            json.dump(p_values, f, indent=4)

        p_values = wilcoxon_test_classifier(grouped_real, column)
        with open(
                os.path.join(RESULTS_DIR, 'stats',
                             f'wilcoxon_{column}_classifier_real.txt'),
                'w') as f:
            json.dump(p_values, f, indent=4)


if __name__ == '__main__':
    main()
