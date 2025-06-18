import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np
from time import sleep

L = 190         # length of CFC in cm
Fa = 2.0 / 60   # wort flow in litres/sec
Fb = 3.0 / 60   # coolant flow in litres/sec
Ta0 = 80.0      # wort input temp
Tb0 = 14.0      # coolant input temp

N = 254         # number of simulation points
dx = L/(N-1)    # space between points

Ca = Fa         # use flow rate as a proxy for heat capacity
Cb = Fb

k = .003 * dx   # thermal conductivity

Ta = np.array(N * [Tb0])    # initialise both circuits to wort temperature
Tb = np.array(N * [Tb0])

# create chart objects globally so we can easily update them
fig, ax = plt.subplots()
plt.grid(axis='y')
Ta_plot = ax.plot(Ta, '-r', label='Wort')[0]
Tb_plot = ax.plot(Tb, '-g', label='Coolant')[0]
Tdiff_plot = ax.plot(Ta - Tb, '-b', label='difference')[0]
ax.set(xlim=[0,N], ylim=[0.0001, 100], ylabel='Temperature')
#ax.set_yscale('log')
ax.legend()

def update_temps():
    for i in range(N):
        # heat flux proportional to temperature difference
        dQ = k * (Ta[i] - Tb[i])

        # update temperatures according to heat capacity
        Ta[i] -= dQ / Ca
        Tb[i] += dQ / Cb

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
    Tdiff_plot.set_ydata(Ta-Tb)
    return (Ta_plot, Tb_plot, Tdiff_plot)

def main():
    ani = animation.FuncAnimation(fig=fig, func=update, save_count=1200, interval=10, blit=True)
    
    # display the animation on the screen
    plt.show()

    # save the animation to file
    #ani.save(filename="./asymptotic_to_coolant_temperature.mp4", fps=60)

    print("Ta", Ta[0], Ta[-1])
    print("Tb", Tb[0], Tb[-1])

if __name__ == '__main__':
    main()