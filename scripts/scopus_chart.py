import os
import sys

import matplotlib.pyplot as plt
from pybliometrics.scopus import ScopusSearch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src import constants

del sys.path[0]


def articles_in_date(year, query):
    search_query = f'TITLE-ABS-KEY({query}) AND PUBYEAR = {year}'
    search = ScopusSearch(search_query, verbose=True, download=False)
    data = search.get_results_size()

    if isinstance(data, dict):
        data = data.get('resultsFound', 0)

    return data


def plot_article_count(start_year, end_year, query):
    years = range(start_year, end_year + 1)
    counts = [articles_in_date(year, query) for year in years]

    plt.figure(figsize=(10, 6))
    plt.plot(years, counts, marker='o')
    plt.title(
        f"Articles about {query} published between {start_year} and {end_year}"
    )
    plt.xlabel("Year")
    plt.ylabel("Number of Articles")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(constants.IMG_DIR + 'scopus_chart.png')


if __name__ == "__main__":
    start_year = 2010
    end_year = 2023
    query = "feature AND selection"
    plot_article_count(start_year, end_year, query)
