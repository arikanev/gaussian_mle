"""Code for testing and visualizing the argmax of a Gaussian PDF"""
import argparse
import bigfloat
import math
import matplotlib.pyplot as plt
import numpy as np


def plt_show():
    try:
        plt.show()
    except UnicodeDecodeError:
        plt_show()


def gaussian_pdf(x, mu, sigma):
    eps = 0
    N = len(x)
    for x_n in x:
        eps += (x_n - mu) ** 2
    log_p = - (N / 2) * math.log(2 * math.pi * sigma**2) - (1 / (2 * sigma**2)) * eps
    p = bigfloat.pow(1 / (2 * math.pi), N / 2) * bigfloat.pow(bigfloat.pow(sigma, 2), N / 2) * bigfloat.exp(- (1 / (2 * sigma ** 2)) * eps)
    return log_p, p


def argmax(x, mu, sigma):
    maxima = -10000000
    points = [[], []]
    for i, mu_param in enumerate(range(50, 0, -1)):
        mu_param /= 10
        for j, sigma_param in enumerate(range(50, 0, -1)):
            sigma_param /= 10
            print("Step {}".format((i+1) * (j+1)))
            log_p, p = gaussian_pdf(x=x, mu=mu_param, sigma=sigma_param)
            print("p_x given mean {} and std dev {} is {}".format(mu_param, sigma_param, log_p))
            points[0].append(log_p)
            points[1].append(p)
            if log_p > maxima:
                maxima = log_p
                mu = mu_param
                sigma = sigma_param

    return mu, sigma, points


def maximum_likelihood_estimation(mu=0, sigma=0.1):

    x = np.random.normal(mu, sigma, 1000)
    # lets find mu, sigma that maximizes x
    # (It should equal the mu, sigma that we've used when defining our distribution)

    N = len(x)

    # maximum likelihood estimate for mu is just sample mean
    mu_hat = (1 / N) * np.sum(x)

    # MLE for sigma is just sqrt of sample variance
    sigma_hat = math.sqrt((1/N) * np.sum((x-mu_hat)**2))

    arg_mean, arg_sigma, points = argmax(x=x, mu=mu, sigma=sigma)

    # fig = plt.figure()
    # ax1 = fig.add_subplot(221)
    # ax2 = fig.add_subplot(222)
    # ax1.scatter([x for x in range(0, len(points[1]))], [p for p in points[1]])
    plt.scatter([x for x in range(0, len(points[0]))], points[0], c=points[0])
    # ax1.title.set_text("Normal distribution")
    plt.title("Log-Normal distribution")
    plt.annotate("(Sample mean {}, sample std_dev {}) & (predicted mean {}, predicted std_dev {})".format(mu_hat, sigma_hat, arg_mean, arg_sigma),
                 (0, 0), (0, -20), xycoords='axes fraction', textcoords='offset points', va='top')
    plt.show()

if __name__ == '__main__':

    # Parse command line arguments.

    parser = argparse.ArgumentParser()

    parser._action_groups.pop()

    required = parser.add_argument_group('required arguments')

    required.add_argument('-mu', '--mean', nargs=1, type=float, default=0,
                          help='set mu of the gaussian')

    required.add_argument('-sig', '--std_dev', nargs=1, type=float, default=1,
                          help='set standard deviation of the gaussian')

    args = parser.parse_args()

    maximum_likelihood_estimation(args.mean, args.std_dev)
