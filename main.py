import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np
from functools import partial

class CFC:
    def __init__(self, Ta_in, Tb_in, Fa, Fb, k, Ca, Cb, N:int):
        self.N = N
        self.Ta_in = Ta_in
        self.Tb_in = Tb_in
        self.Ta = np.array(N * [Ta_in], dtype='float')
        self.Tb = np.array(N * [Tb_in], dtype='float')            
        self.Ca = Fa * Ca
        self.Cb = Fb * Cb
        self.k = k

    def update(self):
        for i in range(self.N):
            dQ = self.k * (self.Ta[i] - self.Tb[i])
            self.Ta[i] -= dQ / self.Ca
            self.Tb[i] += dQ / self.Cb
        self.Ta = np.roll(self.Ta, 1)
        self.Tb = np.roll(self.Tb, -1)
        self.Ta[0] = self.Ta_in
        self.Tb[-1] = self.Tb_in

class Simulation:
    def __init__(self, Ta_in, Tb_in, Fa, Fb, k, Ca:float=1, Cb:float=1, N:int=250):
        self.cfc = CFC(Ta_in, Tb_in, Fa, Fb, k, Ca, Cb, N)
        self.x = np.linspace(0, 1, N)
        self.fig, ax = plt.subplots()
        plt.grid(axis='y')
        self.Ta_plot = ax.plot(self.x, self.cfc.Ta, '-r', label='Wort')[0]
        self.Tb_plot = ax.plot(self.x, self.cfc.Tb, '-g', label='Coolant')[0]
        self.Tdiff_plot = ax.plot(self.x, self.cfc.Ta - self.cfc.Tb, '-b', label='difference')[0]
        ax.set(xlim=[0, 1], ylim=[0, 100], ylabel='Temperature')
        #ax.set_yscale('log')
        ax.legend()

    def update(self, frame):
        self.cfc.update()
        self.Ta_plot.set_ydata(self.cfc.Ta)
        self.Tb_plot.set_ydata(self.cfc.Tb)
        self.Tdiff_plot.set_ydata(self.cfc.Ta - self.cfc.Tb)
        return (self.Ta_plot, self.Tb_plot, self.Tdiff_plot)
    
def main():
    sim = Simulation(N=254, Ta_in=80, Tb_in=14, Fa=2/60, Fb=3/60, k=0.001)
    ani = animation.FuncAnimation(
        fig=sim.fig,
        func=sim.update,
        save_count=1200,
        interval=10,
        blit=True
    )
    plt.show()
    print(sim.cfc.Ta)
    print(sim.cfc.Tb)

if __name__ == '__main__':
    main()