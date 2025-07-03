import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np

class CFC:
    def __init__(self, Ta_in, Tb_in, Fa, Fb, k, Ca, Cb, N:int):
        self.N = N
        self.Ta_in = Ta_in
        self.Tb_in = Tb_in
        self.Ta = np.array(N * [Ta_in], dtype='float')
        self.Tb = np.array(N * [Tb_in], dtype='float')
        self.Ta_out = self.Ta[-1]
        self.Tb_out = self.Tb[0]       
        self.Ca = Fa * Ca
        self.Cb = Fb * Cb
        self.k = k

    def __str__(self):
            strrep = f'Stream A: T_in {self.Ta_in:.3f} T_out {self.Ta_out:.3f} Flow {self.Ca:.3f}\n'
            strrep += f'Stream B: T_in {self.Tb_in:.3f} T_out {self.Tb_out:.3f} Flow {self.Cb:.3f}\n'
            strrep += f'k_q {self.k:.4f}\n'
            return strrep

    def update(self):
        for i in range(self.N):
            dQ = self.k * (self.Ta[i] - self.Tb[i]) / self.N
            self.Ta[i] -= dQ / self.Ca
            self.Tb[i] += dQ / self.Cb
        self.Ta = np.roll(self.Ta, 1)
        self.Tb = np.roll(self.Tb, -1)
        self.Ta[0] = self.Ta_in
        self.Tb[-1] = self.Tb_in
        self.Ta_out = self.Ta[-1]
        self.Tb_out = self.Tb[0]

class Simulation:
    def __init__(self, Ta_in, Tb_in, Fa, Fb, k, Ca:float=1, Cb:float=1, N:int=250):
        self.cfc = CFC(Ta_in, Tb_in, Fa, Fb, k, Ca, Cb, N)
        self.x = np.linspace(0, 1, N)
        self.fig, ax = plt.subplots()
        plt.grid(axis='y')
        self.Ta_plot = ax.plot(self.x, self.cfc.Ta, 'k-', label='Fluid A (flowing right)')[0]
        self.Tb_plot = ax.plot(self.x, self.cfc.Tb, 'k--', label='Fluid B (flowing left)')[0]
        self.Tdiff_plot = ax.plot(self.x, self.cfc.Ta - self.cfc.Tb, 'k:', label='difference')[0]
        self.Ta_model_plot = ax.plot([], [], linestyle='none', marker='o', markeredgecolor='black', markerfacecolor='white', markersize=10, label='Exponential fit A')[0]
        self.Tb_model_plot = ax.plot([], [], linestyle='none', marker='^', markeredgecolor='black', markerfacecolor='white', markersize=10, label='Exponential fit B')[0]
        ax.set(ylim=[0.1, 100], xlabel='Displacement $x$', ylabel='Temperature')
        #ax.set_yscale('log')
        ax.legend(loc='upper right')

    def __str__(self):
        strrep = 'Counterflow\n' + str(self.cfc) + '\n'
        strrep += 'Model\n'
        strrep += f'a {self.a:.3f} b {self.b:.3f} xi {self.xi:.3f} Te {self.Te:.3f}\n'
        return strrep

    def update(self, frame):
        self.cfc.update()
        self.Ta_plot.set_ydata(self.cfc.Ta)
        self.Tb_plot.set_ydata(self.cfc.Tb)
        self.Tdiff_plot.set_ydata(self.cfc.Ta - self.cfc.Tb)

        self.fit_model()
        model_x = np.linspace(0, 1, 6)
        model_Ta = self.Te + self.a * np.exp(-self.xi * model_x)
        model_Tb = self.Te + self.b * np.exp(-self.xi * model_x)
        self.Ta_model_plot.set_xdata(model_x)
        self.Ta_model_plot.set_ydata(model_Ta)
        self.Tb_model_plot.set_xdata(model_x)
        self.Tb_model_plot.set_ydata(model_Tb)

        return (self.Ta_plot, self.Tb_plot, self.Tdiff_plot, self.Ta_model_plot, self.Tb_model_plot)
    
    def fit_model(self):
        cfc = self.cfc
        try:
            self.xi = -np.log((cfc.Ta_out - cfc.Tb_in) / (cfc.Ta_in - cfc.Tb_out))
        except ZeroDivisionError:
            self.xi = 0
        try:
            self.a = (cfc.Ta_out - cfc.Ta_in) * (cfc.Ta_in - cfc.Tb_out) / ((cfc.Ta_out - cfc.Ta_in) + (cfc.Tb_out - cfc.Tb_in))
        except ZeroDivisionError:
            self.a = 0
        try:
            self.b = (cfc.Tb_in - cfc.Tb_out) * (cfc.Ta_in - cfc.Tb_out) / ((cfc.Ta_out - cfc.Ta_in) + (cfc.Tb_out - cfc.Tb_in))
        except ZeroDivisionError:
            self.b = 0
        self.Te = cfc.Ta_in - self.a
    
def main():
    sim = Simulation(N=1000, Ta_in=80, Tb_in=15, Fa=0.025, Fb=0.034, k=0.2)
    ani = animation.FuncAnimation(
        fig=sim.fig,
        func=sim.update,
        save_count=1200,
        interval=10,
        blit=True
    )
    plt.show()
    print(sim)

    print(f'1/tau {sim.cfc.k * (sim.cfc.Ca * sim.cfc.Cb)/(sim.cfc.Ca + sim.cfc.Cb)}')

if __name__ == '__main__':
    main()