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
    # Check if the directory exists
    if os.path.exists(os.path.join(constants.RESULTS_DIR, 'real')):
        result_path_bin = os.path.join(constants.RESULTS_DIR, 'real',
                                       'analysis_results.csv')
        notifications.send_telegram_file(token=token,
                                         chat_id=chat_id,
                                         file_path=result_path_bin,
                                         caption='-- Results binary --',
                                         verbose=False)
    if os.path.exists(os.path.join(constants.RESULTS_DIR, 'real')):
        # If the directory doesn't exist, choose the binary option
        result_path_real = os.path.join(constants.RESULTS_DIR, 'binary',
                                        'analysis_results.csv')
        notifications.send_telegram_file(token=token,
                                         chat_id=chat_id,
                                         file_path=result_path_real,
                                         caption='-- Results real --',
                                         verbose=False)


if __name__ == "__main__":
    main()
