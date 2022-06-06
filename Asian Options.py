#==================================Import the relevant packages=============================#
import numpy as np
from scipy.stats import norm
import matplotlib.pyplot as plt
from scipy import interpolate
#===========================================================================================#
# Function to calculte the Monte Carlo Asian Option Price for Calls or Puts
def Asian_MC(opType,S0,K,T,r,sigma,I):
    M = 250
    dt = T / M
    A = S0 * np.exp(np.cumsum((r - 0.5 * sigma ** 2) * dt
        + sigma * np.sqrt(dt) * np.random.standard_normal((M + 1, I)), axis=0))
    A = A.mean(axis=0)
    if opType == 'call':
        payoff = np.maximum(A-K, 0)
    if opType == 'put':
        payoff = np.maximum(K-A, 0)
    payoff = payoff.mean()    
    payoff = np.exp(-r*T) * payoff
    return payoff
#===========================================================================================#
# Function to calculte the Binomial Tree Asian Option Price for Calls or Puts
    # using equally spaced averages
def Asian_Binomial(opType,S0,K,T,r,sigma):
    M = 100
    dt = T/M # time length of steps
    u = np.exp(sigma*np.sqrt(dt))
    d = 1/u
    p = (np.exp(r*dt)-d)/(u-d)
    mu = np.arange((M+1))
    mu = np.resize(mu,(M+1,M+1)) # sets upper triangular matrix of number of up moves
    md = np.transpose(mu) # sets upper triangular matrix of number of down moves
    mu = u**(mu-md) # sets gross returns of up moves (upper triangular)
    md = d**md # sets gross returns o fdown moves (upper triangular)
    S = S0*mu*md # upper triangular matrix of asset prices (recombining)
    S = np.triu(S, k=0)
    S = np.around(S, 8)
    # Determine the maximum and minimum average of paths progressively along the tree
    ave = np.zeros((M+1, M+1))
    ave[0,0] = S0
    F = np.zeros((M+1, M+1), dtype = object)
    F[0,0] = np.array([S0])
    for i in range(1, M+1):
        ave[0, i] = ((ave[0, i-1] *(i)) + S[0,i]) / (i+1) # maximum average
        ave[i, i] = ((ave[i-1, i-1] *(i)) + S[i,i]) / (i+1) # minimum average
        # Form a grid of averages at each column from the minimum to the maximum average
        arr = np.linspace(ave[i, i], ave[0, i], num=32)
        for j in range(i+1):
            F[j,i] = arr
    # Build the payoff tree    
    payoff = np.zeros((M+1, M+1), dtype=object)
    if opType == 'call':
        terminal = np.maximum(F[0,-1] - K, 0)
    if opType == 'put':
        terminal = np.maximum(K - F[0,-1], 0)
    for i in range(M+1):
        payoff[i,-1] = terminal
    # Backward induction    
    for column in range(M-1, -1, -1):
        for node in range(0, column+1):
            nodelist = []
            for j in range(len(F[node, column])):
                f = F[node, column][j]
                # Determine the new up movement average from that node
                S_ave_up = ((f*(column+1)) + S[node, column+1]) / (column+2)
                # Interpolate up mobement payoff
                model = interpolate.splrep(F[node, column+1], payoff[node, column+1])
                up = interpolate.splev(S_ave_up, model)
                if up < 0: up = 0
                # Determine the new down movement average from that node
                S_ave_down = ((f*(column+1)) + S[node+1, column+1]) / (column+2)
                # Interpolate down movement payoff
                model = interpolate.splrep(F[node+1, column+1], payoff[node+1, column+1])
                down = interpolate.splev(S_ave_down, model)
                if down < 0: down = 0
                # discount the expected payoff
                nodelist.append(np.exp(-r*dt) * (p*up + (1-p)*down))
            nodelist = np.array(nodelist)
            payoff[node, column] = nodelist
    return payoff[0,0][0]
#===========================================================================================#
# Function to calculte the analytic Asian Option Price for Calls or Puts using the
    # log-normal approximation
def Asian_log_normal(S0,K,r,sigma,T):
    # Estimate the first moment
    m1 = ( (np.exp(r*T) - 1) * (S0/r) ) / T
    # Estimate thee second moment
    r1 = r + sigma**2
    r2 = (2*r) + sigma**2
    m2 = (2 * S0**2 * ( ((r*np.exp(r2*T)) - (r2*np.exp(r*T)) + r1) / (r1*r2*r) )) / T
    mu = np.log(m1**2 / np.sqrt(m2))
    v = np.sqrt(np.log(m2/(m1**2)))    
    d1 = (mu - np.log(K) + v**2) / v
    d2 = d1 - v
    c = (norm.cdf(d1) * np.exp(mu-(r*T)+(0.5*v**2))) - (norm.cdf(d2) * (K*np.exp(-r*T)))
    return c
#==========================================================================================#
# Specify the input parameters
opType, S0, K, T, r, sigma, I = 'call', 100, 98, 1, 0.05, 0.15, 100000
#==========================================================================================#
# Calculate the Monte Carlo asian call option price
MC = Asian_MC(opType,S0,K,T,r,sigma,I)
print('The Monte Carlo Asian Call option price is: ', MC)
#==========================================================================================#
# Calculate the binomial tree asian call option price
binomial = Asian_Binomial(opType,S0,K,T,r,sigma)
print('The Binomial Tree Asian Call option price is: ', binomial)
#==========================================================================================#
# Calculate the log_normal approximation asian call option price
log_n = Asian_log_normal(S0,K,r,sigma,T)
print('The Log-Normal Asian Call option price is: ', log_n)
#==========================================================================================#

# Pricing Geometric-Average Options
def bsm(S0, K, T, r, q, sigma, opType):
    d1 = (np.log(S0/K) + (r - q + 0.5*sigma**2) * T) / (sigma*np.sqrt(T))
    d2 = d1 - (sigma*np.sqrt(T))
    c = (S0 * norm.cdf(d1)) - (K*np.exp(-r*T)*norm.cdf(d2))
    if opType=='call':
        return c
    elif opType=='put':
        p = c - S0 + (K*np.exp(-r*T))
        return p
#==========================================================================================#    
    
# continously-sampled Geometric Average-Price call option

def cont_geom_ave_asian(S0, K, T, r, q, sigma, opType):
    V0 = S0 * np.exp( -T * ((6*r + 6*q + sigma**2) / 12)  )
    sigma_avg = sigma / np.sqrt(3)
    prc = bsm(V0, K, T, r, q, sigma_avg, opType)
    return prc

#==========================================================================================#

# discretely-sampled Geometric Average-Price call option

def disc_geom_ave_asian(S0, K, T, r, q, sigma, opType, N):
    dt = T / N
    v = r - q - 0.5*sigma**2
    V0 = S0 * np.exp(-r*T) * np.exp(((N+1)*v*dt*0.5) + ((dt*(N+1)*(2*N+1)*sigma**2) / (12*N)))
    sigma_avg = sigma * (N**-1.5) * np.sqrt(N*(N+1)*(2*N+1)/6)
    prc = bsm(V0, K, T, r, q, sigma_avg, opType)
    return prc
#==========================================================================================#


S0, K, T, r, q, sigma, opType, N = 100, 95, 1, 0.03, 0.0, 0.35, 'call', 200
    
cont = cont_geom_ave_asian(S0, K, T, r, q, sigma, opType) 
disc = disc_geom_ave_asian(S0, K, T, r, q, sigma, opType, N)

print('\nThe price of a continously-sampled geometric-average asian call option is:', round(cont,4))
print('\nThe price of a discretely-sampled geometric-average asian call option is:', round(disc,4))

#==========================================================================================#

