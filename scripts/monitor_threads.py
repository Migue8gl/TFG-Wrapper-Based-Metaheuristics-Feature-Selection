import os
import subprocess
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src import constants, notifications

del sys.path[0]


def check_processes(process_name):
    try:
        # Run pgrep command to check if there are process threads containing the specified string
        subprocess.run(['pgrep', '-f', process_name],
                       check=True,
                       stdout=subprocess.PIPE,
                       stderr=subprocess.PIPE)
        return True  # Process found
    except subprocess.CalledProcessError:
        return False  # Process not found


def main():
    # Notifications
    process_name = "python3 src/main.py"
    python_script = "scripts/group_csv.py"

    while True:
        if not check_processes(process_name):
            # Execute the Python script if no process threads are found
            subprocess.run(['python3', python_script])
            break  # Exit the while loop
        # Wait for 1 minute before checking again
        time.sleep(60)

    token, chat_id = notifications.load_credentials(constants.CREDENTIALS_DIR +
                                                    'credentials.txt')
    notifications.send_telegram_file(token=token,
                                     chat_id=chat_id,
                                     file_path=constants.RESULTS_DIR +
                                     'analysis_results.csv',
                                     caption='Results',
                                     verbose=False)


if __name__ == "__main__":
    main()
