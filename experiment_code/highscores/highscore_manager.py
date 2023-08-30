import yaml
import os

highscore_file = os.path.join(os.path.dirname(__file__), 'highscores.yaml')


def load_highscores():
    """Load the highscores from the file.

    :return: The loaded highscores
    :rtype: dict
    """
    if os.path.exists(highscore_file):
        with open(highscore_file, 'r') as file:
            highscores = yaml.safe_load(file) or {}
    return highscores


def save_highscores(highscores):
    """Save the highscores to the file.

    :param highscores: The highscores to be saved
    :type highscores: dict
    """
    with open(highscore_file, 'w') as file:
        yaml.dump(highscores, file)


def update_highscore(username, game_id, score):
    """Update the highscore for a specific user and game.

    :param username: The username of the player
    :type username: str
    :param game_id: The ID of the game
    :type game_id: str
    :param score: The score achieved by the player
    :type score: int
    :return: The updated highscores for the game
    :rtype: str
    """
    highscores = load_highscores()
    if username not in highscores:
        highscores[username] = {}
    highscores[username][game_id] = score
    save_highscores(highscores)
    return get_highscore_as_string(game_id)


def get_highscore_as_string(game_id):
    """Get the highscores for a specific game as a formatted string.

    :param game_id: The ID of the game
    :type game_id: str
    :return: The highscores for the game as a formatted string
    :rtype: str
    """
    highscores = load_highscores()
    highscore_list = []
    for username in highscores:
        if game_id in highscores[username]:
            highscore_list.append((username, highscores[username][game_id]))

    highscore_list.sort(key=lambda x: x[1], reverse=True)

    formatted_highscores = {}
    for i, (username, score) in enumerate(highscore_list, start=1):
        formatted_highscores[f"{i}. {username}:"] = str(score)

    return formatted_highscores

def check_if_name_exists(username):
    """Check if the username exists in the highscores.

    :param username: The username to check
    :type username: str
    :return: True if the username exists in the highscores, False otherwise
    :rtype: bool
    """
    highscores = load_highscores()
    if username in highscores:
        return True
    else:
        return False
