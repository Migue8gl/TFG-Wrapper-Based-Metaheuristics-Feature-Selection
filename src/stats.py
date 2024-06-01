import os
import sys

import pandas as pd
from scipy.stats import wilcoxon


def main(filename: str):
    # Get dir and file name
    dir, basename = os.path.split(filename)

    # Read excel to DataFrame
    df = pd.read_excel(filename)

    # Create directory to store test results
    if not os.path.exists(os.path.join(dir, 'stats')):
        os.makedirs(os.path.join(dir, 'stats'))

    algorithms = df['alg'].unique()
    df = df.pivot_table(columns='alg')

    p_values = []
    for alg in algorithms:
        p_values_each_alg = []
        for alg2 in algorithms:
            if alg == alg2:
                p_values_each_alg.append(1)
            else:
                p_values_each_alg.append(
                    round(wilcoxon(df[alg], df[alg2], correction=True).pvalue, 4))
        p_values.append(p_values_each_alg)

    df = pd.DataFrame(p_values, index=algorithms, columns=algorithms)

    with pd.ExcelWriter(os.path.join(
            dir, 'stats', f'{str.replace(basename, ".xlsx", "")}_stats.xlsx'),
                        engine='openpyxl') as writer:
        df.to_excel(writer, sheet_name='stats_wilcoxon', index=True)


if __name__ == '__main__':
    main(sys.argv[1])
