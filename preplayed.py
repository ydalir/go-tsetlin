from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import re
import os

from absl import app
import numpy

import pyspiel

NR_OF_MOVES = 100
NR_OF_GAMES = 5000
GAME_DIR = "games/aya_games/"

def _val(s: str):
    return ord(s)-97

def _get_actions(filename: str):
    with open(GAME_DIR+filename) as f:
        content: str = f.read()
        winner = re.search(r"RE\[(\w)", content).group(1)
        winner = 0 if winner == "B" else 1
        moves = re.findall(r"[BW]\[(\w)(\w)\]", content)
        moves = [(_val(B)*19)+_val(W) for (B, W) in moves]
        return [361 if move == 380 else move for move in moves], winner

def _get_action(state, action_str):
    for action in state.legal_actions():
        if str(action_str) == str(action):
            return action

    print("Action not found")
    return None

def play_game(filename: str, game):
    state = game.new_initial_state()
    state = game.new_initial_state()
    moves, result = _get_actions(filename)
    counter = 0
    for move in moves:
        if counter == NR_OF_MOVES:
            break
        counter += 1
        action = _get_action(state, move)
        if str(action) == str(361) and str(prev_action) == str(361):
            break
        prev_action = action
        state.apply_action(action)

    return state.observation_tensor()[:361*2], result

def main(argv):
    filenames = os.listdir(GAME_DIR)
    length = len(filenames)
    game = pyspiel.load_game("go")
    counter = 0

    if os.path.isfile("parsed_games.txt"):
        os.remove("parsed_games.txt")

    with open(f"parsed_games.txt", "a") as f:
        for filename in filenames:
            if counter == NR_OF_GAMES:
                break
            counter += 1
            print(f"{counter} of {length}:\t{filename}")
            tensor, result = play_game(filename, game)
            to_int = [int(x) for x in tensor] + [result]
            printable = " ".join([str(x) for x in to_int]) + "\n"

            f.write(printable)

if __name__ == "__main__":
    app.run(main)