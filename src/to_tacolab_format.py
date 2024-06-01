import os
import sys

import pandas as pd


def main(filename: str):
    directory = os.path.dirname(filename)
    df = pd.read_csv(filename)
    metrics = ['acc', 'avg', 'execution_time', 'selected_rate']

    df['optimizer_classifier'] = df['optimizer'] + '_' + df['classifier']

    for metric in metrics:
        # Pivot the dataframe to get datasets as columns and optimizers as rows
        pivot_df = df.pivot(index='optimizer_classifier',
                            columns='dataset',
                            values=metric)

        # Add a column named 'alg' with the optimizer names
        pivot_df.insert(0, 'alg', pivot_df.index)

        pivot_df.reset_index(drop=True, inplace=True)
        pivot_df = pivot_df.map(lambda x: '{:,.2e}'.format(x)
                                if isinstance(x, float) else x)

        # Renaming the columns from F1 to FN starting from the second column
        pivot_df.columns = ['alg'] + [
            'F{}'.format(i) if isinstance(i, int) else i
            for i in range(1, len(pivot_df.columns))
        ]

        with pd.ExcelWriter(os.path.join(directory,
                                         f'tacolab_results_{metric}.xlsx'),
                            engine='openpyxl') as writer:
            pivot_df.to_excel(writer, sheet_name=metric, index=False)


if __name__ == '__main__':
    main(sys.argv[1])
