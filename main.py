import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np
from time import sleep

L = 190         # length of CFC in cm
Fa = 1.95 / 60  # wort flow in litres/sec
Fb = 2.5 / 60   # coolant flow in litres/sec
Ta0 = 80.0      # wort input temp
Tb0 = 14.0      # coolant input temp

N = 100         # number of simulation points
dx = L/(N-1)    # space between points

Ca = Fa         # use flow rate as a proxy for heat capacity
Cb = Fb

k = 2 * dx   # thermal conductivity

Ta = np.array(N * [Ta0])    # initialise elements of both circuits
Tb = np.array(N * [Tb0])

# create chart objects globally so we can easily update them
fig, ax = plt.subplots()
Ta_plot = ax.plot(Ta, '-r')[0]
Tb_plot = ax.plot(Tb, '-g')[0]
ax.set(ylim=[0,100], ylabel='Temperature')
ax.legend()

def update_temps():
    for i in range(N):
        # heat flux proportional to temperature difference
        dQ = k * (Ta[i] - Tb[i])

        # update temperatures according to heat capacity
        Ta[i] -= dQ * Ca
        Tb[i] += dQ * Cb

    # move everything along one in the direction of flow
    for i in range(N-1, 0, -1):
        Ta[i] = Ta[i-1]
    Ta[0] = Ta0

    for i in range(N-1):
        Tb[i] = Tb[i+1]
    Tb[N-1] = Tb0


def update(frame):
    update_temps()
    Ta_plot.set_ydata(Ta)
    Tb_plot.set_ydata(Tb)
    return (Ta_plot, Tb_plot)

def main():
    ani = animation.FuncAnimation(fig=fig, func=update, frames=40, interval=0)
    plt.show()


if __name__ == '__main__':
    main()