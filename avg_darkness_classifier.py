#### Libraries
# Standard libarary
from collections import defaultdict

# My libraries
import mnist_loader

""" return an average darkness of digit 0 to 9 of training data """
def avg_darkness(training_data):
    digit_counts = defaultdict(int)
    digit_darkness_sum = defaultdict(float)

    for image, digit in zip(training_data[0], training_data[1]):
        digit_counts[digit] += 1
        digit_darkness_sum[digit] += sum(image)

    avgs = defaultdict(float)
    for digit in digit_counts:
        avgs[digit] = digit_darkness_sum[digit] / digit_counts[digit]

    return avgs

""" guess the image based on avgs of training data, return if guess right"""
def guess_digit(image, avgs):
    org_darkness = sum(image)
    darkness_dist = defaultdict(float)
    for (digit, avg_darkness) in avgs.items():
        darkness_dist[digit] = (org_darkness - avg_darkness)**2

    min_dist, guess = float('inf'), -1
    for (digit, dist) in darkness_dist.items():
        if dist<min_dist:
            min_dist = dist
            guess = digit
    return guess

def main():
    # load_data
    training_data, validation_data, test_data = mnist_loader.load_data()
    # based on training_data output avg(darkness) for each digit
    avgs = avg_darkness(training_data)
    # test phase
    guess_right =sum( [ int(guess_digit(image, avgs) == digit) for (image, digit) in zip(test_data[0], test_data[1])])
    # reslt
    print(f"using average darkness: {guess_right} out of {len(test_data[1])} was guessed right")

if __name__=='__main__':
    main()