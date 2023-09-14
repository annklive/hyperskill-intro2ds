# write your code here
N = 100
data = ""
while True:
    print("Print a random string containing 0 or 1:")
    random_string = input()
    data = data + ''.join([x for x in random_string if x == '0' or x == '1'])
    if len(data) < N:
        print(f"Current data length is {len(data)}, {N-len(data)} symbols left")
    else:
        break
print("Final data string:")
print(data)

for n in range(8):
    triad = bin(n)[2:].zfill(3)
    counts = [0, 0]
    start_pos = 0
    while True:
        spot = data.find(triad, start_pos)
        if spot == -1 or spot+3 > len(data)-1: 
            break
        binary = int(data[spot+3])
        counts[binary] += 1
        start_pos = spot + 1

    print("%s: %d,%d" % (triad, counts[0], counts[1]))
