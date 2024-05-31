import json
import os

import pandas as pd
from scipy.stats import levene, shapiro, ttest_ind, wilcoxon

from .constants import RESULTS_DIR


def check_normality(data):
    _, p_value = shapiro(data)
    return p_value


def check_homogeneity(var1, var2):
    _, p_value = levene(var1, var2)
    return p_value


def apply_student_t_test(var1, var2):
    _, p_value = ttest_ind(var1, var2)
    return p_value


def apply_welch_t_test(var1, var2):
    _, p_value = ttest_ind(var1, var2, equal_var=False)
    return p_value


def apply_wilcoxon_test(var1, var2):
    _, p_value = wilcoxon(var1, var2)
    return p_value


def main():
    df_binary = pd.read_csv(RESULTS_DIR + 'binary/analysis_results.csv')
    df_real = pd.read_csv(RESULTS_DIR + 'real/analysis_results.csv')

    p_values = {}
    if not os.path.exists(os.path.join(RESULTS_DIR, 'stats')):
        os.makedirs(os.path.join(RESULTS_DIR, 'stats'))

    for column in ['avg', 'acc', 'selected_rate', 'execution_time']:
        grouped_binary = df_binary.groupby('optimizer')
        grouped_real = df_real.groupby('optimizer')

        for optimizer, group in grouped_binary:
            if optimizer != 'aco':
                avg_opt1 = group[column]
                avg_opt2 = grouped_real.get_group(optimizer)[column]

                normality_p_value = check_normality(avg_opt1)
                homogeneity_p_value = check_homogeneity(avg_opt1, avg_opt2)

                if normality_p_value > 0.05 and homogeneity_p_value > 0.05:
                    p_value = apply_student_t_test(avg_opt1, avg_opt2)
                    test = 't-test'
                elif normality_p_value > 0.05:
                    p_value = apply_welch_t_test(avg_opt1, avg_opt2)
                    test = 'welch-t-test'
                else:
                    p_value = apply_wilcoxon_test(avg_opt1, avg_opt2)
                    test = 'wilcoxon-test'
                p_values[optimizer] = p_value

        with open(
                os.path.join(RESULTS_DIR, 'stats',
                             f'{test}_{column}_optimizer.txt'), 'w') as f:
            json.dump(p_values, f, indent=4)


if __name__ == '__main__':
    main()
