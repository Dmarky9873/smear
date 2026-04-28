import random
from engine import Game, get_legal_actions
from models import Play
from constants import SUITS


def get_player_names(num_players: int) -> list[str]:
    player_names = list()
    for i in range(num_players):
        name = input(f"player {i + 1} name (leave blank for number): ")
        if name == "":
            name = f"{i + 1}"
        while name in player_names:
            name = input(
                f"that player name is already taken, please choose another: ")
            if name == "":
                name = f"{i + 1}"
        player_names.append(name)

    return player_names


def get_teams(players: list) -> set[tuple[str, str]]:
    p = players.copy()
    teams = set()
    while p != []:
        team = input(
            "enter a team of two people (names, comma-seperated): ").split(',')
        for i, _ in enumerate(team):
            team[i] = team[i].strip()
        if len(team) != 2:
            print("Invalid team length. Please enter a team of two.")
        elif any(player not in p for player in team):
            print(
                "Invalid player entered. Please enter player names that aren't already in teams.")
        else:
            teams.add(tuple(team))
            for player in team:
                p.remove(player)
    return teams


def main():
    num_players = int(input("how many players: "))
    player_names = get_player_names(num_players)
    is_teams = input("teams? (y/n): ").lower()
    if is_teams not in {'y', 'n'}:
        raise ValueError(
            f"answer yes or no to wanting teams. you answered {is_teams}")
    is_teams = is_teams == "y" and num_players % 2 == 0
    if is_teams:
        teams = get_teams(player_names)
    else:
        teams = {(player) for player in player_names}

    game = Game(num_players, player_names, teams)

    return game


if __name__ == "__main__":
    main()
