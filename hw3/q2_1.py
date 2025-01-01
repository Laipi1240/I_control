import numpy as np
import matplotlib.pyplot as plt
import skfuzzy as fuzz
from skfuzzy import control as ctrl

# Define the fuzzy logic controller for cruise control
def setup_fuzzy_controller():
    # Define fuzzy variables with adjusted ranges
    error = ctrl.Antecedent(np.arange(-10, 11, 0.1), 'error')  # Speed error (target speed - current speed)
    derivative = ctrl.Antecedent(np.arange(-2, 2.1, 0.1), 'derivative')  # Rate of change of error
    control_signal = ctrl.Consequent(np.arange(-10, 11, 0.1), 'control_signal')  # Throttle control (bounded between -10 and 10)

    # Define membership functions for error
    error['NB'] = fuzz.trimf(error.universe, [-10, -10, -5])
    error['NS'] = fuzz.trimf(error.universe, [-10, -5, 0])
    error['Z'] = fuzz.trimf(error.universe, [-2.5, 0, 2.5])
    error['PS'] = fuzz.trimf(error.universe, [0, 5, 10])
    error['PB'] = fuzz.trimf(error.universe, [5, 10, 10])

    # Define membership functions for derivative (rate of change of error)
    derivative['NB'] = fuzz.trimf(derivative.universe, [-2, -2, -1])
    derivative['NS'] = fuzz.trimf(derivative.universe, [-2, 0, 2])
    derivative['Z'] = fuzz.trimf(derivative.universe, [-1, 0, 1])
    derivative['PS'] = fuzz.trimf(derivative.universe, [0, 1, 2])
    derivative['PB'] = fuzz.trimf(derivative.universe, [1, 2, 2])

    # Define membership functions for control signal (throttle control)
    control_signal['NB'] = fuzz.trimf(control_signal.universe, [-10, -10, -5])
    control_signal['NS'] = fuzz.trimf(control_signal.universe, [-10, -5, 0])
    control_signal['Z'] = fuzz.trimf(control_signal.universe, [-5, 0, 5])
    control_signal['PS'] = fuzz.trimf(control_signal.universe, [0, 5, 10])
    control_signal['PB'] = fuzz.trimf(control_signal.universe, [5, 10, 10])

    # Define fuzzy rules
    rule1 = ctrl.Rule(error['NB'] & derivative['NB'], control_signal['NB'])
    rule2 = ctrl.Rule(error['NB'] & derivative['NS'], control_signal['NB'])
    rule3 = ctrl.Rule(error['NB'] & derivative['Z'], control_signal['NS'])
    rule4 = ctrl.Rule(error['NB'] & derivative['PS'], control_signal['Z'])
    rule5 = ctrl.Rule(error['NB'] & derivative['PB'], control_signal['Z'])
    rule6 = ctrl.Rule(error['NS'] & derivative['NB'], control_signal['NB'])
    rule7 = ctrl.Rule(error['NS'] & derivative['NS'], control_signal['NS'])
    rule8 = ctrl.Rule(error['NS'] & derivative['Z'], control_signal['NS'])
    rule9 = ctrl.Rule(error['NS'] & derivative['PS'], control_signal['Z'])
    rule10 = ctrl.Rule(error['NS'] & derivative['PB'], control_signal['Z'])
    rule11 = ctrl.Rule(error['Z'] & derivative['NB'], control_signal['NS'])
    rule12 = ctrl.Rule(error['Z'] & derivative['NS'], control_signal['NS'])
    rule13 = ctrl.Rule(error['Z'] & derivative['Z'], control_signal['Z'])
    rule14 = ctrl.Rule(error['Z'] & derivative['PS'], control_signal['PS'])
    rule15 = ctrl.Rule(error['Z'] & derivative['PB'], control_signal['PS'])
    rule16 = ctrl.Rule(error['PS'] & derivative['NB'], control_signal['Z'])
    rule17 = ctrl.Rule(error['PS'] & derivative['NS'], control_signal['Z'])
    rule18 = ctrl.Rule(error['PS'] & derivative['Z'], control_signal['PS'])
    rule19 = ctrl.Rule(error['PS'] & derivative['PS'], control_signal['PB'])
    rule20 = ctrl.Rule(error['PS'] & derivative['PB'], control_signal['PB'])
    rule21 = ctrl.Rule(error['PB'] & derivative['NB'], control_signal['Z'])
    rule22 = ctrl.Rule(error['PB'] & derivative['NS'], control_signal['Z'])
    rule23 = ctrl.Rule(error['PB'] & derivative['Z'], control_signal['PS'])
    rule24 = ctrl.Rule(error['PB'] & derivative['PS'], control_signal['PB'])
    rule25 = ctrl.Rule(error['PB'] & derivative['PB'], control_signal['PB'])

    # Create the control system
    control_system = ctrl.ControlSystem([
        rule1, rule2, rule3, rule4, rule5,
        rule6, rule7, rule8, rule9, rule10,
        rule11, rule12, rule13, rule14, rule15,
        rule16, rule17, rule18, rule19, rule20,
        rule21, rule22, rule23, rule24, rule25])

    fuzzy_controller = ctrl.ControlSystemSimulation(control_system)
    return fuzzy_controller

# Fuzzy control simulation
def fuzzy_control(fuzzy_controller, error, derivative):
    fuzzy_controller.input['error'] = error
    fuzzy_controller.input['derivative'] = derivative
    fuzzy_controller.compute()
    return fuzzy_controller.output['control_signal']

# Simulation parameters for cruise control system
dt = 0.1          # Simulation time step
time_end = 50     # End time for the simulation
t = np.arange(0, time_end, dt)

# Set-point for the vehicle's target speed (cruise control target speed)
target_speed = 20  # Target speed in m/s (e.g., 20 m/s = 72 km/h)

# Simulate the vehicle response with fuzzy control
def simulate_vehicle_fuzzy(t, target_speed):
    fuzzy_controller = setup_fuzzy_controller()
    n = len(t)
    error = np.zeros(n)  # Speed error (target - actual)
    derivative = 0        # Derivative of error
    speed = np.zeros(n)   # Vehicle speed (output of the system)
    
    T_m = 190
    w_n = 420
    alpha_n = 25
    beta = 0.4
    rho = 1.3
    C_d = 0.32
    C_r = 0.01
    A = 2.4
    m = 1500
    g = 9.8

    # Loop through each time step to simulate the system
    for i in range(1, n):
        error[i] = target_speed - speed[i-1]  # Calculate error
        if i > 1:
            derivative = (error[i] - error[i-1]) / dt  # Calculate derivative of error
        
        # Fuzzy control
        control_signal = fuzzy_control(fuzzy_controller, error[i], derivative)
        
        # Apply the control signal (throttle) to adjust vehicle speed
        T = T_m*(1-beta*(alpha_n*speed[i-1]/w_n-1)**2)
        F_e = control_signal*T*alpha_n
        F_d = 0.5*rho*C_d*A*speed[i-1]**2
        F_r = m*g*C_r*(1 if speed[i-1] > 0 else -1)
        speed[i] = speed[i-1] + (F_e-F_d-F_r)/m
    
    return speed

# Simulate and plot the response with the fuzzy controller
response = simulate_vehicle_fuzzy(t, target_speed)

# Plot the results
plt.plot(t, np.ones_like(t) * target_speed, 'r--', label="Target Speed")
plt.plot(t, response, 'b', label="Actual Speed")
plt.xlabel("Time (s)")
plt.ylabel("Speed (m/s)")
plt.title("Vehicle Speed with Fuzzy Cruise Control")
plt.legend()
plt.grid()
plt.show()
