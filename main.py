'''
Simulates a counterflow heat exchanger using finite elements,
fits an exponential model to the input and output teperatures
and infers the conductivity of the heat exchanger from the model
'''

import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np

class Flow:
    def __init__(self, length, mass, mass_rate, specific_heat_capacity, Tin, flow_direction='R', N:int=256):
        self.N = N
        self.flow_direction = flow_direction
        self.element_length = length / N
        self.element_mass = mass / N
        self.mass_rate = mass_rate
        self.element_heat_cap = specific_heat_capacity * self.element_mass
        self.Tin = Tin
        self.T = np.array(N * [Tin], dtype='float')

    @property
    def Tout(self):
        if self.flow_direction == 'R':
            Tout = self.T[self.N - 1]
        elif self.flow_direction == 'L':
            Tout = self.T[0]
        else:
            raise ValueError('unrecognised flow direction')
        return Tout

    def __str__(self):
        strrep = f'N {self.N} dx {self.element_length} dm {self.element_mass} dC {self.element_heat_cap} '
        strrep += f'mass flow {self.mass_rate} ({self.flow_direction}) Tin {self.Tin} Tout {self.Tout}\n'
        strrep += f'   {self.T}\n'
        return strrep
    
    def timestep(self, dt):
        weight_Tin = self.mass_rate * dt / self.element_mass
        weight_Told = 1 - weight_Tin
        assert weight_Told >= 0, f'dt too high for mass flow rate (max {self.element_mass / self.mass_rate})'
        if self.flow_direction == 'R':
            for n in range(self.N-1, -1, -1):
                if n > 0:
                    Tin = self.T[n-1]
                else:
                    Tin = self.Tin
                Told = self.T[n]
                self.T[n] = Tin * weight_Tin + Told * weight_Told
        elif self.flow_direction == 'L':
            for n in range(self.N):
                if n < (self.N - 1):
                    Tin = self.T[n+1]
                else:
                    Tin = self.Tin
                Told = self.T[n]
                self.T[n] = Tin * weight_Tin + Told * weight_Told
        else:
                raise ValueError('unrecognised flow direction')

class CFC:
    def __init__(self, length, kq, num_elements,
                 mass_a, mass_rate_a, specific_heat_capacity_a, Tin_a,
                 mass_b, mass_rate_b, specific_heat_capacity_b, Tin_b):
        self.t = 0
        self.length = length
        self.kq = kq
        self.N = num_elements
        self.a = Flow(length, mass_a, mass_rate_a, specific_heat_capacity_a, Tin_a, 'R', num_elements)
        self.b = Flow(length, mass_b, mass_rate_b, specific_heat_capacity_b, Tin_b, 'L', num_elements)

    def __str__(self):
        strrep = f'L {self.length} kq {self.kq}\n'
        strrep += f'a: {self.a}'
        strrep += f'b: {self.b}'
        return strrep
    
    def heat_transfer(self, dt):
        for n in range(self.N):
            dQab = self.kq / self.N * (self.a.T[n] - self.b.T[n]) * dt
            self.a.T[n] -= dQab / self.a.element_heat_cap
            self.b.T[n] += dQab / self.b.element_heat_cap
    
    def timestep(self, dt):
        self.a.timestep(dt)
        self.b.timestep(dt)
        self.heat_transfer(dt)
        self.t += dt
    

class Simulation:
    def __init__(self, length, kq, num_elements, time_step,
                 mass_a, mass_rate_a, specific_heat_capacity_a, Tin_a,
                 mass_b, mass_rate_b, specific_heat_capacity_b, Tin_b):
        self.cfc = CFC(
            length=length, kq=kq, num_elements=num_elements,
            mass_a=mass_a, mass_rate_a=mass_rate_a, specific_heat_capacity_a=specific_heat_capacity_a, Tin_a=Tin_a,
            mass_b=mass_b, mass_rate_b=mass_rate_b, specific_heat_capacity_b=specific_heat_capacity_b, Tin_b=Tin_b)
        self.dt = time_step

        # set up the chart
        self.x = np.linspace(0, length, num_elements)
        self.fig, ax = plt.subplots()
        plt.grid(axis='y')
        self.Ta_plot = ax.plot(self.x, self.cfc.a.T, 'k-', label='Fluid A (flowing right)')[0]
        self.Tb_plot = ax.plot(self.x, self.cfc.b.T, 'k--', label='Fluid B (flowing left)')[0]
        self.Tdiff_plot = ax.plot(self.x, self.cfc.a.T - self.cfc.b.T, 'k:', label='difference')[0]
        self.Ta_model_plot = ax.plot([], [], linestyle='none', marker='o', markeredgecolor='black', markerfacecolor='white', markersize=10, label='Exponential fit A')[0]
        self.Tb_model_plot = ax.plot([], [], linestyle='none', marker='^', markeredgecolor='black', markerfacecolor='white', markersize=10, label='Exponential fit B')[0]
        ax.set(ylim=[0.1, 100], xlabel='Displacement $x$', ylabel='Temperature')
        #ax.set_yscale('log')
        ax.legend(loc='upper right')

    def __str__(self):
        strrep = f'Counterflow: dt {self.dt} {self.cfc}\n'
        strrep += f'Model: a {self.a:.4f} b {self.b:.4f} xi {1.0/self.kx:.4f} Te {self.Te:.4f}\n'
        return strrep

    def update(self, frame):
        self.cfc.timestep(self.dt)
        self.Ta_plot.set_ydata(self.cfc.a.T)
        self.Tb_plot.set_ydata(self.cfc.b.T)
        self.Tdiff_plot.set_ydata(self.cfc.a.T - self.cfc.b.T)

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
            self.kx = -np.log((cfc.a.Tout - cfc.b.Tin) / (cfc.a.Tin - cfc.b.Tout))
            self.a = (cfc.a.Tout - cfc.a.Tin) * (cfc.a.Tin - cfc.b.Tout) / ((cfc.a.Tout - cfc.a.Tin) + (cfc.b.Tout - cfc.b.Tin))
            self.b = (cfc.b.Tin - cfc.b.Tout) * (cfc.a.Tin - cfc.b.Tout) / ((cfc.a.Tout - cfc.a.Tin) + (cfc.b.Tout - cfc.b.Tin))
        except ZeroDivisionError:
            self.kx = 0
            self.a = 0
            self.b = 0
        self.Te = cfc.a.Tin - self.a

        
def main():
    np.set_printoptions(precision=4, floatmode='fixed', threshold=10)

    #! this is where we set the parameters for the simulation
    sim = Simulation(length=1, kq=1, time_step=0.0005, num_elements=512,
                     mass_a=1, mass_rate_a=2, specific_heat_capacity_a=1, Tin_a=80,
                     mass_b=1, mass_rate_b=0.75, specific_heat_capacity_b=1, Tin_b=20)
    
    ani = animation.FuncAnimation(
        fig=sim.fig,
        func=sim.update,
        save_count=1200,
        interval=0,
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
    Ta_0 = sim.cfc.a.Tin
    Ta_1 = sim.cfc.a.Tout
    Tb_0 = sim.cfc.b.Tout
    Tb_1 = sim.cfc.b.Tin

    # temp differences at either end of the CFC
    deltaT_0 = Ta_0 - Tb_0
    deltaT_1 = Ta_1 - Tb_1

    # heat capacity flow rates (product of mass flow rate and specific heat capacity)
    Ca = sim.cfc.a.element_heat_cap / sim.cfc.a.element_mass * sim.cfc.a.mass_rate
    Cb = sim.cfc.b.element_heat_cap / sim.cfc.b.element_mass * sim.cfc.b.mass_rate

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