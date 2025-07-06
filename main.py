'''
Simulates a counterflow heat exchanger using finite elements,
fits an exponential model to the input and output teperatures
and infers the conductivity of the heat exchanger from the model
'''

import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np

class CFC:
    def __init__(self, Ta_in, Tb_in, Fa, Fb, kq, Ca, Cb, N:int):
        self.N = N
        self.Ta_in = Ta_in
        self.Tb_in = Tb_in
        self.Ta = np.array(N * [Ta_in], dtype='float')
        self.Tb = np.array(N * [Tb_in], dtype='float')
        self.Ta_out = self.Ta[-1]
        self.Tb_out = self.Tb[0]       
        self.Ca = Fa * Ca   # adjust the heat capacities so that we
        self.Cb = Fb * Cb   # can pretend the flow rates are equal
        self.kq = kq

    def __str__(self):
            strrep = f'Stream A: T_in {self.Ta_in:.4f} T_out {self.Ta_out:.4f} Flow {self.Ca:.4f}\n'
            strrep += f'Stream B: T_in {self.Tb_in:.4f} T_out {self.Tb_out:.4f} Flow {self.Cb:.4f}\n'
            strrep += f'kq {self.kq:.4f}\n'
            return strrep

    def update(self):
        for i in range(self.N):
            dQ = self.kq * (self.Ta[i] - self.Tb[i]) / (self.N - 1)
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
        strrep += f'a {self.a:.4f} b {self.b:.4f} xi {1.0/self.kx:.4f} Te {self.Te:.4f}\n'
        return strrep

    def update(self, frame):
        self.cfc.update()
        self.Ta_plot.set_ydata(self.cfc.Ta)
        self.Tb_plot.set_ydata(self.cfc.Tb)
        self.Tdiff_plot.set_ydata(self.cfc.Ta - self.cfc.Tb)

        self.fit_model()
        model_x = np.linspace(0, 1, 6)
        model_Ta = self.Te + self.a * np.exp(-self.kx * model_x)
        model_Tb = self.Te + self.b * np.exp(-self.kx * model_x)
        self.Ta_model_plot.set_xdata(model_x)
        self.Ta_model_plot.set_ydata(model_Ta)
        self.Tb_model_plot.set_xdata(model_x)
        self.Tb_model_plot.set_ydata(model_Tb)

        return (self.Ta_plot, self.Tb_plot, self.Tdiff_plot, self.Ta_model_plot, self.Tb_model_plot)
    
    def fit_model(self):
        # fit exponential models to input and output temps, so that
        #   Ta = Te + a e^(-kx x)
        #   Tb = Te + b e^(-kx x)
        cfc = self.cfc
        try:
            self.kx = -np.log((cfc.Ta_out - cfc.Tb_in) / (cfc.Ta_in - cfc.Tb_out))
            self.a = (cfc.Ta_out - cfc.Ta_in) * (cfc.Ta_in - cfc.Tb_out) / ((cfc.Ta_out - cfc.Ta_in) + (cfc.Tb_out - cfc.Tb_in))
            self.b = (cfc.Tb_in - cfc.Tb_out) * (cfc.Ta_in - cfc.Tb_out) / ((cfc.Ta_out - cfc.Ta_in) + (cfc.Tb_out - cfc.Tb_in))
        except ZeroDivisionError:
            self.kx = 0
            self.a = 0
            self.b = 0
        self.Te = cfc.Ta_in - self.a
    
def main():
    #! this line is where we set the parameters for the simulation
    sim = Simulation(N=1024, Ta_in=80, Tb_in=20, Fa=2, Fb=0.75, k=0.5)
    ani = animation.FuncAnimation(
        fig=sim.fig,
        func=sim.update,
        save_count=1200,
        interval=10,
        blit=True
    )
    plt.show()  # simulation will run until the user closes the MatPlotLib window

    # display CFC and model parameters
    print(sim)

    # fitted model parameters
    a = sim.a
    b = sim.b
    kx = sim.kx

    # derivatives of fitted exponentials 
    #   dTa/dx = (-a kx) e^(-kx x)
    #   dTb/dx = (-b kx) e^(-kx x)
    dTadx_0 = -a * kx
    dTadx_1 = -a * kx * np.exp(-kx)
    dTbdx_0 = -b * kx
    dTbdx_1 = -b * kx * np.exp(-kx)

    # temps at either end of the CFC
    Ta_0 = sim.cfc.Ta_in
    Ta_1 = sim.cfc.Ta_out
    Tb_0 = sim.cfc.Tb_out
    Tb_1 = sim.cfc.Tb_in

    # temp differences at either end of the CFC
    deltaT_0 = Ta_0 - Tb_0
    deltaT_1 = Ta_1 - Tb_1

    # heat capacity flow rates (product of mass flow rate and specific heat capacity)
    Ca = sim.cfc.Ca
    Cb = sim.cfc.Cb

    # infer the CFC conductivity from the HC rates, dT/dx and delta T
    kq_a0 = -Ca * dTadx_0 / deltaT_0
    kq_a1 = -Ca * dTadx_1 / deltaT_1
    kq_b0 = -Cb * dTbdx_0 / deltaT_0
    kq_b1 = -Cb * dTbdx_1 / deltaT_1

    calc_kq = (kq_a0 + kq_a1 + kq_b0 + kq_b1) / 4.0

    print('Calculated kq')
    print(f'dTa/dx {dTadx_0:.4f}, {dTadx_1:.4f}')
    print(f'dTb/dx {dTbdx_0:.4f}, {dTbdx_1:.4f}')
    print(f'deltaT {deltaT_0:.4f}, {deltaT_1:.4f}')
    print(f'kq_a0 {kq_a0:.4f}\nkq_a1 {kq_a1:.4f}\nkq_b0 {kq_b0:.4f}\nkq_b1 {kq_b1:.4f}')
    print(f'kq_mean {calc_kq:.4f}')

     # calculate the spatial distibution parameters
    calc_xi = -1/(calc_kq * (Ca - Cb)/(Ca * Cb))
    calc_a = 1/(1 - Ca/Cb * np.exp(-1/calc_xi)) * (Ta_0 - Tb_1)
    calc_b = 1/(Cb/Ca - np.exp(-1/calc_xi)) * (Ta_0 - Tb_1)
    calc_Te = Ta_0 - calc_a

    print('\nCalculated spatial parmeters')
    print(f'a {calc_a:.4f} b {calc_b:.4f} xi {calc_xi:.4f} Te {calc_Te:.4f}')

if __name__ == '__main__':
    main()