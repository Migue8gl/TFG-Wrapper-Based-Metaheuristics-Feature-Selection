import datetime
import subprocess
from collections import defaultdict

import constants
import matplotlib.pyplot as plt


def get_commit_data() -> list:
    """
    Returns a list of commit dates in the format "YYYY-MM".

    Returns:
        commit_data (list): A list of commit dates in the format "YYYY-MM".
    """
    command = 'git log --pretty=format:"%ad" --date=format:"%Y-%m" --reverse'
    result = subprocess.run(command,
                            shell=True,
                            capture_output=True,
                            text=True)
    commit_data = result.stdout.splitlines()
    return commit_data


# Function to process commit data and count commits per month
def count_commits(commit_data: list) -> dict:
    """
    Returns a dictionary of commit counts per month.
    
    Args:
        commit_data (list): A list of commit dates in the format "YYYY-MM".
    
    Returns:
        commits_per_month (dict): A dictionary of commit counts per month.
    """
    commits_per_month = defaultdict(int)
    for commit_date in commit_data:
        month = datetime.datetime.strptime(commit_date,
                                           "%Y-%m").strftime("%Y-%m")
        commits_per_month[month] += 1
    return commits_per_month


# Function to plot the data
def plot_data(data: dict):
    """
    Plots the data as a line chart.

    Args:
        data (dict): A dictionary of commit counts per month.
    """
    months = list(data.keys())
    commits = list(data.values())

    plt.figure(figsize=(10, 6))
    plt.plot(months,
             commits,
             marker='o',
             color='skyblue',
             linewidth=2,
             markersize=8)
    plt.fill_between(months, commits, color='skyblue', alpha=0.2)
    plt.xlabel('Month')
    plt.ylabel('Number of Commits')
    plt.title('Commits per Month')
    plt.xticks(rotation=45)
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(constants.IMG_DIR + 'commits_chart.png')


if __name__ == "__main__":
    commit_data = get_commit_data()
    commits_per_month = count_commits(commit_data)
    plot_data(commits_per_month)
