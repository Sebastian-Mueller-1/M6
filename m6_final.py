import numpy as np
from scipy.integrate import odeint
import matplotlib.pyplot as plt
import random

iSize = 45
jSize = 80

# Inital Population Counts, for N can just read CSV data into needed 2d data array in one step
N = np.genfromtxt("BEE529 M6 Dataset NApop.csv", delimiter=',')
I0 = np.zeros((iSize,jSize));
R0 = np.zeros((iSize,jSize));

# define random initialization function to find infected individual start point.
def random_init():
    initial_x = random.randint(0,79) 
    initial_y = random.randint(0,44)
    return [initial_y, initial_x]

# define loop preventing checking init population to make sure infected individual from starting in the ocean
attempt = random_init()
while N[attempt[0]][attempt[1]] == 0.0:
    attempt = random_init()

I0[attempt[0]][attempt[1]] = 1 # assign one infected individual 
S0 = N - I0 - R0

# find total population from CSV data
nMax = 0.0
for row in N:
    for value in row:
        nMax+=value


beta = 0.33
gamma = 1./10 
print('r_0 is', beta/gamma)
# Migration Rate [people per day of the total population per boundary]
percent_migrate = .001
# A grid of time points (in days)
simulation_days = 1095
dt = 0.1
t = np.linspace(0, simulation_days, int(simulation_days/dt))

# Empty Output Location
yOut = np.zeros((iSize,jSize,len(t),3))

# The SIR model differential equations.
def spatialSIR(y):
    # Current System Status
    S = y[0]; I = y[1]; R = y[2]
    # Empty Derivitive Terms
    dSdt = np.copy(S)*0; 
    dIdt = np.copy(I)*0;
    dRdt = np.copy(R)*0;
    
    # loop through all locations
    for i in np.arange(S.shape[0]):
        for j in np.arange(S.shape[1]):
        
            if N[i,j] != 0: # check not on ocean cell
            
                # internal contribution
                dSdt[i,j] = (-beta*S[i,j]*I[i,j]/N[i,j]) 
                dIdt[i,j] = (beta*S[i,j]*I[i,j]/N[i,j] - gamma*I[i,j])
                dRdt[i,j] = (gamma*I[i,j])

                # find valid boundaries and calculate change contribution of immigration and emigration
                if i>0 and N[i-1,j] != 0: # north check
                    #find smallest population to determine migration rate
                    smallest_pop = None 
                    if N[i,j] < N[i-1,j]:
                        smallest_pop = N[i,j]
                    else:
                        smallest_pop = N[i-1,j]
                    
                    #set migration rate
                    mrate = percent_migrate*smallest_pop

                    #immigration contribution
                    dSdt[i,j] += mrate*(S[i-1,j]/N[i-1,j])
                    dIdt[i,j] += mrate*(I[i-1,j]/N[i-1,j])
                    dRdt[i,j] += mrate*(R[i-1,j]/N[i-1,j])

                    #emigration technique
                    dSdt[i,j] -= mrate*(S[i,j]/N[i,j])
                    dIdt[i,j] -= mrate*(I[i,j]/N[i,j])
                    dRdt[i,j] -= mrate*(R[i,j]/N[i,j])

                if j>0 and N[i,j-1] != 0: #west check 
                    #find smallest population to determine migration rate
                    smallest_pop = None 
                    if N[i,j] < N[i,j-1]:
                        smallest_pop = N[i,j]
                    else:
                        smallest_pop = N[i,j-1]
                    
                    #set migration rate
                    mrate = percent_migrate*smallest_pop

                    #immigration contribution
                    dSdt[i,j] += mrate*(S[i,j-1]/N[i,j-1])
                    dIdt[i,j] += mrate*(I[i,j-1]/N[i,j-1])
                    dRdt[i,j] += mrate*(R[i,j-1]/N[i,j-1])

                    #emigration technique
                    dSdt[i,j] -= mrate*(S[i,j]/N[i,j])
                    dIdt[i,j] -= mrate*(I[i,j]/N[i,j])
                    dRdt[i,j] -= mrate*(R[i,j]/N[i,j])      

                if i<44 and N[i+1,j] != 0: #south check 
                    #find smallest population to determine migration rate
                    smallest_pop = None 
                    if N[i,j] < N[i+1,j]:
                        smallest_pop = N[i,j]
                    else:
                        smallest_pop = N[i+1,j]
                    
                    #set migration rate
                    mrate = percent_migrate*smallest_pop

                    #immigration contribution
                    dSdt[i,j] += mrate*(S[i+1,j]/N[i+1,j])
                    dIdt[i,j] += mrate*(I[i+1,j]/N[i+1,j])
                    dRdt[i,j] += mrate*(R[i+1,j]/N[i+1,j])

                    #emigration technique
                    dSdt[i,j] -= mrate*(S[i,j]/N[i,j])
                    dIdt[i,j] -= mrate*(I[i,j]/N[i,j])
                    dRdt[i,j] -= mrate*(R[i,j]/N[i,j])      

                if j<79 and N[i,j+1] != 0: #east check 
                    #find smallest population to determine migration rate
                    smallest_pop = None 
                    if N[i,j] < N[i,j+1]:
                        smallest_pop = N[i,j]
                    else:
                        smallest_pop = N[i,j+1]
                    
                    #set migration rate
                    mrate = percent_migrate*smallest_pop

                    #immigration contribution
                    dSdt[i,j] += mrate*(S[i,j+1]/N[i,j+1])
                    dIdt[i,j] += mrate*(I[i,j+1]/N[i,j+1])
                    dRdt[i,j] += mrate*(R[i,j+1]/N[i,j+1])

                    #emigration technique
                    dSdt[i,j] -= mrate*(S[i,j]/N[i,j])
                    dIdt[i,j] -= mrate*(I[i,j]/N[i,j])
                    dRdt[i,j] -= mrate*(R[i,j]/N[i,j])     

    # Get next value
    Sout = S + dSdt*dt
    Iout = I + dIdt*dt
    Rout = R + dRdt*dt
    return [Sout, Iout, Rout]

# Set initial Conditions
yOut[:,:,0,0] = S0; yOut[:,:,0,1] = I0; yOut[:,:,0,2] = R0;

# Iterate into future
for curt in np.arange(1,len(t)):
    y0 = [yOut[:,:,curt-1,0], yOut[:,:,curt-1,1], yOut[:,:,curt-1,2]]
    [yOut[:,:,curt,0], yOut[:,:,curt,1], yOut[:,:,curt,2]] = spatialSIR(y0)


# Set Up Axes
from matplotlib import animation
fig, axs = plt.subplots(1, 3,figsize=(12, 3))
cax0 = axs[0].pcolormesh(np.log(np.flipud(yOut[:, :, 0, 0])))
cax1 = axs[1].pcolormesh(np.log(np.flipud(yOut[:, :, 0, 1])))
cax2 = axs[2].pcolormesh(np.log(np.flipud(yOut[:, :, 0, 2])))
fig.colorbar(cax2)
fig.colorbar(cax1)
fig.colorbar(cax0)

# What to Plot at i
def animate(i):
    cax0.set_array(np.log(np.flipud(yOut[:, :, i,0])))
    cax1.set_array(np.log(np.flipud(yOut[:, :, i,1])))
    cax2.set_array(np.log(np.flipud(yOut[:, :, i,2])))
    axs[0].set_title('S at %d days' %t[i])
    axs[1].set_title('I at %d days' %t[i])
    axs[2].set_title('R at %d days' %t[i])
    plt.tight_layout()

# Make the Animation
anim = animation.FuncAnimation(fig, animate, frames=np.arange(0, len(t), 50), interval = 10)
gif_name = 'SIR_Final_1095_run4.gif'
anim.save(gif_name)

# Display the GIF
from IPython.display import Image
Image(url=gif_name) 
    
plt.figure(1)
S1 = yOut[attempt[0],attempt[1],:,0]; I1 = yOut[attempt[0],attempt[1],:,1]; R1 = yOut[attempt[0],attempt[1],:,2]
plt.plot(t, S1/1000, 'b', label='Susceptible')
plt.plot(t, I1/1000, 'r', label='Infected')
plt.plot(t, R1/1000, 'g', label='Recovered with immunity')
plt.xlabel('Time /days')
plt.ylabel('Number (1000s)')
plt.legend(loc='center right')

plt.figure(1)
S1 = np.sum(yOut[:,:,:,0],(0,1)); I1 = np.sum(yOut[:,:,:,1],(0,1)); R1 = np.sum(yOut[:,:,:,2],(0,1))
plt.plot(t, S1/1E+6, 'b', label='Susceptible')
plt.plot(t, I1/1E+6, 'r', label='Infected')
plt.plot(t, R1/1E+6, 'g', label='Recovered with immunity')
plt.xlabel('Time /days')
plt.ylabel('Number (millions)')
plt.legend(loc='center right')
