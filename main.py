import matplotlib.pyplot as plt
import numpy as np
from time import sleep

L = 190         # length of CFC in cm
Fa = 1.95 / 60  # wort flow in litres/sec
Fb = 2.5 / 60   # coolant flow in litres/sec
Ta0 = 80.0      # wort input temp
Tb0 = 14.0      # coolant input temp

N = 10          # number of simulation points
dx = L/(N-1)    # space between points

Ca = Fa         # use flow rate as a proxy for heat capacity
Cb = Fb

k = 0.01 * dx   # thermal conductivity

Ta = np.array(N * [Ta0])
Tb = np.array(N * [Tb0])

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



def main():

    for i in range(100):
        update_temps()

        print(Ta.round(3))
        print(Tb.round(3))

        print()


if __name__ == '__main__':
    main()