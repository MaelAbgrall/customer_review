import sys


def progressBar(value, endvalue, bar_length=20):
    """print a progress bar

    Arguments:
        value {int} -- current value
        endvalue {int} -- maximum value
    """

    percent = float(value) / endvalue
    arrow = '-' * int(round(percent * bar_length)-1) + '>'
    spaces = ' ' * (bar_length - len(arrow))

    sys.stdout.write("\r[{0}] {1}%".format(
        arrow + spaces, int(round(percent * 100))))
    sys.stdout.flush()
