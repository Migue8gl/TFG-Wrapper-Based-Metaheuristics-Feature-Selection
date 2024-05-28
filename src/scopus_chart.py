from typing import Optional

import constants
import matplotlib.pyplot as plt
from pybliometrics.scopus import ScopusSearch


def articles_in_date(year: int, query: str) -> int:
    """
    Retrieves the number of articles for a given year and query.

    Parameters:
        - year (int): The year to search for.
        - query (str): The search query.

    Returns:
        - data (int): The number of articles found.
    """
    search_query = f'TITLE-ABS-KEY({query}) AND PUBYEAR = {year}'
    search = ScopusSearch(search_query, verbose=True, download=False)
    data = search.get_results_size()

    if isinstance(data, dict):
        data = data.get('resultsFound', 0)

    return data


def plot_article_count(start_year: int,
                       end_year: int,
                       query: str,
                       img_name: Optional[str] = 'scopus_chart.png'):
    """
    Plots the number of articles over the specified years.

    Parameters:
        - start_year (int): The starting year.
        - end_year (int): The ending year.
        - query (str): The search query.
        - img_name (str, optional): The name of the image file to save. Defaults to 'scopus_chart.png'.
    """
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
    plt.savefig(constants.IMG_DIR + img_name)


if __name__ == "__main__":
    start_year = 2010
    end_year = 2023
    query = "feature AND selection"
    img_name = "scopus_chart.png"

    plot_article_count(start_year, end_year, query, img_name)

    query = "feature AND selection AND metaheuristics"
    img_name = "scopus_chart2.png"
    plot_article_count(start_year, end_year, query, img_name)

    algorithms = [
        'Grey Wolf Optimizer', 'Grasshopper Optimization Algorithm',
        'Firefly Algorithm', 'Cuckoo Search', 'Whale Optimization Algorithm',
        'Bat Algorithm', 'Dragonfly Algorithm'
    ]

    for algorithm in algorithms:
        img_name = f"scopus_chart_{algorithm.replace(' ', '_')}.png"
        query = algorithm.replace(' ', ' AND ')
        plot_article_count(start_year, end_year, query, img_name)
