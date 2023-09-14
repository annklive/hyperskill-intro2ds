import random

print("Please provide AI some data to learn...\n"
      "The current data length is 0, 100 symbols left")
N = 100
data = ""
while True:
    print("Print a random string containing 0 or 1:\n")
    random_string = input()
    data = data + ''.join([x for x in random_string if x == '0' or x == '1'])
    if len(data) < N:
        print(f"The current data length is {len(data)}, {N - len(data)} symbols left")
    else:
        break
print("\nFinal data string:")
print(data)

triad_counts = {}
for n in range(8):
    triad = bin(n)[2:].zfill(3)
    counts = [0, 0]
    start_pos = 0
    while True:
        spot = data.find(triad, start_pos)
        if spot == -1 or spot + 3 > len(data) - 1:
            break
        binary = int(data[spot + 3])
        counts[binary] += 1
        start_pos = spot + 1
    triad_counts[triad] = counts

print('\nYou have $1000. Every time the system successfully predicts your next press, you lose $1.\n'
      'Otherwise, you earn $1. Print "enough" to leave the game. Let\'s go!\n')

remaining_credits = 1000
min_test_data_length = 4
end_game = False
while True:
    test_data = ''
    while len(test_data) < min_test_data_length:
        test_string = input('Print a random string containing 0 or 1:\n')
        if test_string == 'enough':
            end_game = True
            break
        test_data = ''.join([b for b in test_string if b == '0' or b == '1'])

    if not end_game:
        predictions = ''
        idx = 0
        correct = 0
        total = 0
        while idx < len(test_data) - 3:
            triad = test_data[idx:idx + 3]
            if triad_counts[triad][0] > triad_counts[triad][1]:
                predictions += '0'
            elif triad_counts[triad][0] < triad_counts[triad][1]:
                predictions += '1'
            else:
                predictions += str(random.randint(0, 2))

            if predictions[-1] == test_data[idx + 3]:
                correct += 1
                remaining_credits -= 1
            else:
                remaining_credits += 1

            idx += 1

        print('predictions:')
        print(predictions)

        print(f'\n\nComputer guessed {correct} out of {idx} symbols right ({correct * 100 / idx:.2f} %)')
        print(f'Your balance is now ${remaining_credits}')

    else:
        print('Game over!')
        break

