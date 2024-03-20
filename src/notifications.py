from typing import Union

import requests


def load_credentials(filename: str) -> tuple:
    """
    Load Telegram bot credentials from a text file.

    Parameters:
        - filename (str): Name of the text file containing credentials.

    Returns:
        - tuple: A tuple containing the token and chat ID.
    """
    with open(filename, 'r') as file:
        token = file.readline().strip()
        chat_id = file.readline().strip()
    return token, chat_id


def send_telegram_message(token: str,
                          chat_id: str,
                          message: str = 'Test',
                          verbose: bool = False) -> Union[dict, str]:
    """
    Send a message through Telegram.

    Parameters:
        - token (str): Your Telegram bot token.
        - chat_id (str): Your Telegram chat ID.
        - message (str): The message to send.
        - verbose (bool): If True, print the JSON response. Default is False.

    Returns:
        - Union[dict, str]: The JSON response from the Telegram API or an error message as a string.
    """
    url = f"https://api.telegram.org/bot{token}/sendMessage"
    params = {'chat_id': chat_id, 'text': message}

    try:
        response = requests.get(url, params=params)
        response.raise_for_status()
        json_response = response.json()
        if verbose:
            print(json_response)
        return json_response
    except requests.RequestException as e:
        error_message = {'error': f"Failed to send message: {e}"}
        if verbose:
            print(error_message)
        return error_message


def send_telegram_image(token: str,
                        chat_id: str,
                        image_path: str = None,
                        caption: str = 'Test',
                        verbose: bool = False) -> Union[dict, str]:
    """
    Send an image through Telegram.

    Parameters:
        - token (str): Your Telegram bot token.
        - chat_id (str): Your Telegram chat ID.
        - image_path (str): Path to the image file.
        - caption (str): Caption for the image. Default is None.
        - verbose (bool): If True, print the JSON response. Default is False.

    Returns:
        - Union[dict, str]: The JSON response from the Telegram API or an error message as a string.
    """
    url = f"https://api.telegram.org/bot{token}/sendPhoto"
    params = {
        'chat_id': chat_id,
        'caption': caption
    } if caption else {
        'chat_id': chat_id
    }

    try:
        with open(image_path, 'rb') as image_file:
            files = {'photo': image_file}
            response = requests.post(url, params=params, files=files)
            response.raise_for_status()
            json_response = response.json()
            if verbose:
                print(json_response)
            return json_response
    except requests.RequestException as e:
        error_message = {'error': f"Failed to send image: {e}"}
        if verbose:
            print(error_message)
        return error_message


def send_telegram_file(token: str,
                       chat_id: str,
                       file_path: str = None,
                       caption: str = None,
                       verbose: bool = False) -> Union[dict, str]:
    """
    Send a file through Telegram.

    Parameters:
        - token (str): Your Telegram bot token.
        - chat_id (str): Your Telegram chat ID.
        - file_path (str): Path to the file.
        - caption (str): Caption for the file. Default is None.
        - verbose (bool): If True, print the JSON response. Default is False.

    Returns:
        - Union[dict, str]: The JSON response from the Telegram API or an error message as a string.
    """
    url = f"https://api.telegram.org/bot{token}/sendDocument"
    params = {
        'chat_id': chat_id,
        'caption': caption
    } if caption else {
        'chat_id': chat_id
    }

    try:
        with open(file_path, 'rb') as file:
            files = {'document': file}
            response = requests.post(url, params=params, files=files)
            response.raise_for_status()
            json_response = response.json()
            if verbose:
                print(json_response)
            return json_response
    except requests.RequestException as e:
        error_message = {'error': f"Failed to send file: {e}"}
        if verbose:
            print(error_message)
        return error_message


if __name__ == "__main__":
    # Load credentials from file
    token, chat_id = load_credentials('./credentials/credentials.txt')
    # Send a message
    send_telegram_message(token, chat_id, message='Hola, test 1', verbose=True)
