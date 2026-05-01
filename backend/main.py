try:
    from .engine import Game
    from .simulator import Simulator
except ImportError:
    from engine import Game
    from simulator import Simulator


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
    ...


if __name__ == "__main__":
    main()
