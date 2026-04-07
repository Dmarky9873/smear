from engine import Game


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


def main():
    num_players = int(input("how many players: "))
    player_names = get_player_names(num_players)

    game = Game(num_players, player_names)

    game.init_new_game()


if __name__ == "__main__":
    main()
