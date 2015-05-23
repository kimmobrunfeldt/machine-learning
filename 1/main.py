import random
import json
from sklearn.tree import DecisionTreeClassifier, export_graphviz
import numpy as np


def run_game():
    games = []

    for x in xrange(1000):
        coin = random.randint(0, 1)
        bet = random.randint(0, 1)

        # coin 1, bet 1 = double your bet
        profit = 2 * bet * coin
        games.append([coin, bet, profit])

    return games


def calculate_accuracy(dataset, correct):
    correct_len = len(filter(lambda x: x == correct, dataset))
    return '%s%%' % (float(correct_len) / len(dataset) * 100)


def main():
    data = run_game()

    clf = DecisionTreeClassifier(criterion='entropy')

    game_data = [[i[0], i[1]] for i in data]
    profits = [i[2] for i in data]

    clf.fit(game_data, profits)

    with open('tree.dot', 'w') as dotfile:
        export_graphviz(
            clf,
            dotfile,
            feature_names=['coin', 'bet']
        )

    predictions_lose1 = [clf.predict([0, 0]) for x in xrange(100)]
    predictions_lose2 = [clf.predict([0, 1]) for x in xrange(100)]
    predictions_win = [clf.predict([1, 1]) for x in xrange(100)]

    print 'All these profit predictions should be zero:'
    print predictions_lose1
    print 'Accuracy was', calculate_accuracy(predictions_lose1, np.array([0]))

    print 'All these profit predictions should be zero:'
    print predictions_lose2
    print 'Accuracy was', calculate_accuracy(predictions_lose2, np.array([0]))

    print 'All these profit predictions should be two:'
    print predictions_win
    print 'Accuracy was', calculate_accuracy(predictions_win, np.array([2]))


if __name__ == '__main__':
    main()
