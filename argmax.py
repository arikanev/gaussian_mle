"""Code for testing and visualizing the argmax of a Gaussian PDF"""
import bigfloat
import math
import matplotlib.pyplot as plt
import numpy as np

mu, sigma = 0, 0.1
x = np.random.normal(mu, sigma, 1000)


def plt_show():
    try:
        plt.show()
    except UnicodeDecodeError:
        plt_show()


def gaussian_pdf(x=x, mu=mu, sigma=sigma):
    eps = 0
    N = len(x)
    for x_n in x:
        eps += (x_n - mu) ** 2
    log_p = - (N / 2) * math.log(2 * math.pi * sigma**2) - (1 / (2 * sigma**2)) * eps
    p = bigfloat.pow(1 / (2 * math.pi), N / 2) * bigfloat.pow(bigfloat.pow(sigma, 2), N / 2) * bigfloat.exp(- (1 / (2 * sigma ** 2)) * eps)
    return log_p, p


def argmax():
    maxima = -10000000
    points = [[], []]
    for i, mu_param in enumerate(range(50, 0, -1)):
        mu_param /= 10
        for j, sigma_param in enumerate(range(50, 0, -1)):
            sigma_param /= 10
            print("Step {}".format((i+1) * (j+1)))
            log_p, p = gaussian_pdf(mu=mu_param, sigma=sigma_param)
            print("p_x given mean {} and std dev {} is {}".format(mu_param, sigma_param, log_p))
            points[0].append(log_p)
            points[1].append(p)
            if log_p > maxima:
                maxima = log_p
                mu = mu_param
                sigma = sigma_param
    fig = plt.figure()
    ax1 = fig.add_subplot(221)
    ax2 = fig.add_subplot(222)
    ax1.scatter([x for x in range(0, len(points[1]))], [p for p in points[1]])
    ax2.scatter([x for x in range(0, len(points[0]))], points[0])
    ax1.title.set_text("Normal distribution")
    ax2.title.set_text("Log-Normal distribution")
    plt.show()
    return mu, sigma


# lets find mu, sigma that maximizes x
# (It should equal the mu, sigma that we've used when defining our distribution)

N = len(x)

# maximum likelihood estimate for mu is just sample mean
mu_hat = (1 / N) * np.sum(x)

# MLE for sigma is just sqrt of sample variance
sigma_hat = math.sqrt((1/N) * np.sum((x-mu_hat)**2))

arg_mean, arg_sigma = argmax()

print("(Sample mean {}, sample std_dev {}) & (predicted mean {}, predicted std_dev {})".format(mu_hat, sigma_hat, arg_mean, arg_sigma))

