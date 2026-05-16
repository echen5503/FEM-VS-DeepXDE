L = 1.0           # Length of the domain (m)
k = 50.0          # Thermal conductivity (W/m·K)
Q = 10000.0     # Uniform heat source (W/m³)
T0 = 100.0        # Boundary condition: Temperature at x=0 (°C)
TL = 50.0         # Boundary condition: Temperature at x=L (°C)

# Transient parameters
rho = 7850.0      # Density (kg/m³)
c_p = 500.0       # Specific heat capacity (J/kg·K)
T_init = 20.0     # Initial uniform temperature (°C)
t_end = 100000.0   # Total simulation time (s)
dt = 100.0        # Time step size (s)
