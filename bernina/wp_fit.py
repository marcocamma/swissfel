import numpy as np
import matplotlib.pyplot as plt
plt.ion()
import lmfit

theta, energy = np.loadtxt("wp_cal.txt", unpack=True)
plt.plot(theta,energy,'o',label='exp')
plt.xlabel("$\Theta (\deg)$")
plt.ylabel("Energy per pulse (mJ)")

def wp(params,theta):
    energy0 = params['energy0'].value
    theta0 = params['theta0'].value
    scale = params['scale'].value
    energy_min = params['energy_min'].value
    return energy0*np.cos(scale*np.deg2rad(theta-theta0))**2+energy_min

def myfunc(params,theta,energy):
    energy_calc = wp(params,theta)
    residual_array = energy - energy_calc
    return residual_array

def fit(pars=None):
    fit_params = lmfit.Parameters()
    if pars==None:
      fit_params.add('energy0',value=2.,min=1.5,max=5.)
      fit_params.add('theta0',value=203.5,min=200.,max=210.)
      fit_params.add('scale',value=2.,vary=False)
      fit_params.add('energy_min',value=0.0,min=-0.05,max=0.05)
    else:
      fit_params.add('energy0',value=pars[0],min=1.5,max=2.5)
      fit_params.add('theta0',value=pars[1],min=180.,max=220.)
      fit_params.add('scale',value=pars[2],min=0.5,max=2.)
      fit_params.add('energy_min',value=pars[3],min=-0.1,max=0.1)
    result = lmfit.minimize(myfunc, fit_params, args=(theta, energy))
    print("chisqr = %.3f" % result.chisqr)
    return result

def plot_fit():
    result = fit()
    theta_fit = theta
    energy_fit = wp(result.params,theta_fit)
    plt.plot(theta_fit,energy_fit,label='fit')
    plt.legend()

def from_energy_to_wp(energy,result=None):
    if result is None:
        print("Error: provide result of fit as second parameter")
    else:
        energy0 = result.params['energy0'].value
        theta0 = result.params['theta0'].value
        scale = result.params['scale'].value
        energy_min = result.params['energy_min'].value
        delta_theta = np.arccos(np.sqrt((energy-energy_min)/energy0))/2.
        delta_theta = np.rad2deg(delta_theta)
        print(delta_theta)
        theta = delta_theta + theta0
        return theta

result = fit()

