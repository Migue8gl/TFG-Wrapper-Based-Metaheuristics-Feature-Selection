import subprocess
from collections import defaultdict
import matplotlib.pyplot as plt
import datetime

# Function to extract commit data
def get_commit_data():
    command = 'git log --pretty=format:"%ad" --date=format:"%Y-%m" --reverse'
    result = subprocess.run(command, shell=True, capture_output=True, text=True)
    commit_data = result.stdout.splitlines()
    return commit_data

# Function to process commit data and count commits per month
def count_commits(commit_data):
    commits_per_month = defaultdict(int)
    for commit_date in commit_data:
        month = datetime.datetime.strptime(commit_date, "%Y-%m").strftime("%Y-%m")
        commits_per_month[month] += 1
    return commits_per_month

# Function to plot the data
def plot_data(data):
    months = list(data.keys())
    commits = list(data.values())

    plt.figure(figsize=(10, 6))
    plt.plot(months, commits, marker='o', color='skyblue', linewidth=2, markersize=8)
    plt.fill_between(months, commits, color='skyblue', alpha=0.2)
    plt.xlabel('Month')
    plt.ylabel('Number of Commits')
    plt.title('Commits per Month')
    plt.xticks(rotation=45)
    plt.grid(True)
    plt.tight_layout()
    plt.savefig('images/commits_per_month.png')
    plt.show()

if __name__ == "__main__":
    commit_data = get_commit_data()
    commits_per_month = count_commits(commit_data)
    plot_data(commits_per_month)
