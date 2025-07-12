'''
Simulates a counterflow heat exchanger using finite elements,
fits an exponential model to the input and output teperatures
and infers the conductivity of the heat exchanger from the model
'''

import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np

class Flow:
    def __init__(self, length, total_mass, mass_flow_rate, specific_heat_capacity, Tin, flow_direction='R', N:int=256):
        self.N = N
        self.flow_direction = flow_direction
        self.element_length = length / N
        self.element_mass = total_mass / N
        self.mass_flow_rate = mass_flow_rate
        self.specific_heat_capacity = specific_heat_capacity
        self.element_heat_capacity = specific_heat_capacity * self.element_mass
        self.Tfeed = Tin
        self.T = np.array(N * [Tin], dtype='float')

    @property
    def Tin(self):
        if self.flow_direction == 'R':
            Tin = self.T[0]
        elif self.flow_direction == 'L':
            Tin = self.T[self.N - 1]
        else:
            raise ValueError('unrecognised flow direction')
        return Tin

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
        strrep = f'N {self.N} dx {self.element_length} dm {self.element_mass} dC {self.element_heat_capacity} '
        strrep += f'mass flow {self.mass_flow_rate} ({self.flow_direction}) Tin {self.Tin} Tout {self.Tout}\n'
        strrep += f'   {self.T}\n'
        return strrep
    
    def timestep(self, dt):
        Tnew_proportion = self.mass_flow_rate * dt / self.element_mass
        Told_proportion = 1 - Tnew_proportion
        if Told_proportion < 0:
            raise ValueError (f'dt too high for mass flow rate, max {self.element_mass / self.mass_flow_rate})')
        if self.flow_direction == 'R':
            for n in range(self.N-1, -1, -1):
                if n > 0:
                    Tnew = self.T[n-1]
                else:
                    Tnew = self.Tfeed
                Told = self.T[n]
                self.T[n] = Tnew * Tnew_proportion + Told * Told_proportion
        elif self.flow_direction == 'L':
            for n in range(self.N):
                if n < (self.N - 1):
                    Tnew = self.T[n+1]
                else:
                    Tnew = self.Tfeed
                Told = self.T[n]
                self.T[n] = Tnew * Tnew_proportion + Told * Told_proportion
        else:
                raise ValueError('unrecognised flow direction')

class CFC:
    def __init__(self, length, kq, num_elements,
                 total_mass_a, mass_flow_rate_a, specific_heat_capacity_a, Tin_a,
                 total_mass_b, mass_flow_rate_b, specific_heat_capacity_b, Tin_b):
        self.t = 0
        self.length = length
        self.kq = kq
        self.N = num_elements
        self.a = Flow(length, total_mass_a, mass_flow_rate_a, specific_heat_capacity_a, Tin_a, 'R', num_elements)
        self.b = Flow(length, total_mass_b, mass_flow_rate_b, specific_heat_capacity_b, Tin_b, 'L', num_elements)
        self.element_kq = kq * self.a.element_length

    def __str__(self):
        strrep = f'L {self.length} kq {self.kq}\n'
        strrep += f'a: {self.a}'
        strrep += f'b: {self.b}'
        return strrep
    
    def heat_transfer(self, dt):
        for n in range(self.N):
            dQab = self.element_kq * (self.a.T[n] - self.b.T[n]) * dt
            self.a.T[n] -= dQab / self.a.element_heat_capacity
            self.b.T[n] += dQab / self.b.element_heat_capacity
    
    def timestep(self, dt):
        self.a.timestep(dt)
        self.b.timestep(dt)
        self.heat_transfer(dt)        
        self.t += dt
    

class Simulation:
    def __init__(self, length, kq, num_elements, mass_new_old_ratio,
                 total_mass_a, mass_flow_rate_a, specific_heat_capacity_a, Tin_a,
                 total_mass_b, mass_flow_rate_b, specific_heat_capacity_b, Tin_b):
        self.cfc = CFC(
            length=length, kq=kq, num_elements=num_elements,
            total_mass_a=total_mass_a, mass_flow_rate_a=mass_flow_rate_a, specific_heat_capacity_a=specific_heat_capacity_a, Tin_a=Tin_a,
            total_mass_b=total_mass_b, mass_flow_rate_b=mass_flow_rate_b, specific_heat_capacity_b=specific_heat_capacity_b, Tin_b=Tin_b)
        
        # calculate the time step to give the required new/old mass ratio for the smallest element
        dt_a = mass_new_old_ratio * self.cfc.a.element_mass / self.cfc.a.mass_flow_rate
        dt_b = mass_new_old_ratio * self.cfc.b.element_mass / self.cfc.b.mass_flow_rate
        self.dt = min(dt_a, dt_b)

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
        strrep = f'Counterflow: dt {self.dt:.4f} {self.cfc}\n'
        strrep += f'Model: a {self.a:.4f} b {self.b:.4f} xi {self.xi:.4f} Te {self.Te:.4f}\n'
        return strrep

    def fit_model(self):
        # fit exponential models to input and output temps, so that
        #   Ta = Te + a e^(-x/xi)
        #   Tb = Te + b e^(-x/xi)
        cfc = self.cfc
        np.seterr(divide='raise')
        try:
            self.xi = -self.cfc.length / np.log((cfc.a.Tout - cfc.b.Tin) / (cfc.a.Tin - cfc.b.Tout))
            self.a = (cfc.a.Tout - cfc.a.Tin) * (cfc.a.Tin - cfc.b.Tout) / ((cfc.a.Tout - cfc.a.Tin) + (cfc.b.Tout - cfc.b.Tin))
            self.b = (cfc.b.Tin - cfc.b.Tout) * (cfc.a.Tin - cfc.b.Tout) / ((cfc.a.Tout - cfc.a.Tin) + (cfc.b.Tout - cfc.b.Tin))
        except FloatingPointError:
            self.xi = 1
            self.a = 0
            self.b = 0
        self.Te = cfc.a.Tin - self.a

    def update(self, frame):
        self.cfc.timestep(self.dt)
        self.Ta_plot.set_ydata(self.cfc.a.T)
        self.Tb_plot.set_ydata(self.cfc.b.T)
        self.Tdiff_plot.set_ydata(self.cfc.a.T - self.cfc.b.T)

        self.fit_model()
        model_x = np.linspace(0, self.cfc.length, 6)
        model_Ta = self.Te + self.a * np.exp(-model_x/self.xi)
        model_Tb = self.Te + self.b * np.exp(-model_x/self.xi)
        self.Ta_model_plot.set_xdata(model_x)
        self.Ta_model_plot.set_ydata(model_Ta)
        self.Tb_model_plot.set_xdata(model_x)
        self.Tb_model_plot.set_ydata(model_Tb)

        return (self.Ta_plot, self.Tb_plot, self.Tdiff_plot, self.Ta_model_plot, self.Tb_model_plot)

        
def main():
    np.set_printoptions(precision=4, floatmode='fixed', threshold=10)

    #! this is where we set the parameters for the simulation
    sim = Simulation(length=2, kq=1.247665, num_elements=256, mass_new_old_ratio=0.2,
                     total_mass_a=1, mass_flow_rate_a=0.45, specific_heat_capacity_a=1, Tin_a=80,
                     total_mass_b=1, mass_flow_rate_b=0.6, specific_heat_capacity_b=1, Tin_b=15)
    
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
    xi = sim.xi
    L = sim.cfc.length
    kq = sim.cfc.kq

    # temps at either end of the CFC
    Ta_in = sim.cfc.a.Tin
    Ta_out = sim.cfc.a.Tout
    Tb_in = sim.cfc.b.Tin
    Tb_out = sim.cfc.b.Tout

    # heat capacity flow rates (product of mass flow rate and specific heat capacity)
    Ca_rate = sim.cfc.a.mass_flow_rate * sim.cfc.a.specific_heat_capacity
    Cb_rate = sim.cfc.b.mass_flow_rate * sim.cfc.b.specific_heat_capacity

    kq_a = 1/L * Ca_rate * (Ta_out-Ta_in)/(Ta_out-Ta_in+Tb_out-Tb_in) * np.log((Ta_in-Tb_out)/(Ta_out-Tb_in))
    kq_b = 1/L * Cb_rate * (Tb_in-Tb_out)/(Ta_out-Ta_in+Tb_out-Tb_in) * np.log((Ta_in-Tb_out)/(Ta_out-Tb_in))

    kq_mean = (kq_a + kq_b) / 2

    print('Calculated kq')
    print(f'kq_a {kq_a:.4f}\nkq_b {kq_b:.4f}\nkq_mean {kq_mean:.4f}')
    xi_new = 1/( 1/L * np.log((Ta_in-Tb_out)/(Ta_out-Tb_in)) )
    xi_new2 = 1/( kq * (1/Ca_rate - 1/Cb_rate) )
    print(f'xi_new {xi_new} {xi_new2}')
    

     # calculate the spatial distibution parameters
    calc_xi = -1/(kq_mean * (Ca_rate - Cb_rate)/(Ca_rate * Cb_rate))
    calc_a = 1/(1 - Ca_rate/Cb_rate * np.exp(-L/calc_xi)) * (Ta_in - Tb_in)
    calc_b = 1/(Cb_rate/Ca_rate - np.exp(-L/calc_xi)) * (Ta_in - Tb_in)
    calc_Te = Ta_in - calc_a

    print('\nCalculated spatial parameters')
    print(f'a {calc_a:.4f} b {calc_b:.4f} xi {calc_xi:.4f} Te {calc_Te:.4f}')

if __name__ == '__main__':
    main()