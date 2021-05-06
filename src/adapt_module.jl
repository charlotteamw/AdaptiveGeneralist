using Parameters
using LinearAlgebra
using ForwardDiff
using PyPlot
using DifferentialEquations
using NLsolve
using Statistics
using RecursiveArrayTools
using Noise

## Here we will illustrate the case of a generalist predator that ultimately has a climate-driven differential response to its prey. In this case, we change from 
# 20C-30C and show a generalist predator switching between prey in alternate habitats. Here, the generalist predator is omnivorous, and has different temperature responses in different habitats (littoral & pelagic)


## Parameters are categorized by macrohabitat -> parameters with "_litt" indicate littoral macrohabitat values and those with "_pel" indicate pelagic macrohabitat values  

## Parameter alitt in model is the temperature dependent attack rate of P on C_litt and apel is the temperature dependent attack rate of P on C_Pel. All other attack rates are not temp dependent

@with_kw mutable struct AdaptPar
    
    r_litt = 1.0
    r_pel = 1.0
    α_pel = 0.6    ##competitive influence of pelagic resource on littoral resource 
    α_litt = 0.6   ## competitve influence of littoral resource on pelagic resource
    k_litt = 1.0 
    k_pel = 1.0
    h_CR = 0.5
    h_PC = 0.5
    h_PR = 0.5
    e_CR = 0.8
    e_PC = 0.8
    e_PR = 0.8
    m_C = 0.2
    m_P = 0.3
    a_CR_litt = 0.8
    a_CR_pel = 0.8
    a_PR_litt = 0.1 
    a_PR_pel = 0.1
    aT_litt = 1.0
    aT_pel = 1.0
    Tmax_litt = 40
    Topt_litt = 32
    Tmax_pel = 32
    Topt_pel = 25
    σ = 6
    T = 20
    noise = 0
    
end


## Omnivory Module with Temp Dependent Attack Rates (alitt => aPC in littoral zone; apel => aPC in pelagic zone)

function adapt_model!(du, u, p, t)
    @unpack r_litt, r_pel, k_litt, k_pel, α_pel, α_litt, e_CR, e_PC, e_PR, aT_pel, aT_litt, a_CR_litt, a_CR_pel, a_PR_litt, a_PR_pel, h_CR, h_PC, h_PR, m_C, m_P, T, Topt_litt, Tmax_litt, aT_litt, Topt_pel, Tmax_pel, aT_pel, σ = p 
    
    a_PC_litt = ifelse(T < Topt_litt,  
    aT_litt * exp(-((T - Topt_litt)/(2*σ))^2), 
    aT_litt * (1 - ((T - (Topt_litt))/((Topt_litt) - Tmax_litt))^2)
    )
    
    a_PC_pel = ifelse(T < Topt_pel, 
    aT_pel * exp(-((T - Topt_pel)/(2*σ))^2),
    aT_pel * (1 - ((T - (Topt_pel))/((Topt_pel) - Tmax_pel))^2) 
    )
    
    R_litt, R_pel, C_litt, C_pel, P = u
    
    du[1] = (r_litt * R_litt) * (1 - ((α_pel * R_litt + R_pel)/k_litt)) - ((a_CR_litt * R_litt * C_litt)/ (1+ a_CR_litt + h_CR + R_litt )) - ((a_PR_litt * R_litt * P)/ (1 + (a_PR_litt * h_PR * R_litt) + (a_PR_pel * h_PR  * R_pel) + (a_PC_litt * h_PC * C_litt) + (a_PC_pel * h_PC * C_pel))) 
    du[2] = (r_pel * R_pel) * (1 - ((α_litt * R_litt + R_pel)/k_pel)) - ((a_CR_pel * R_pel * C_pel)/ (1+ a_CR_pel + h_CR + R_pel )) - ((a_PR_pel * R_pel * P)/ (1 + (a_PR_litt * h_PR * R_litt) + (a_PR_pel * h_PR  * R_pel) + (a_PC_litt * h_PC * C_litt) + (a_PC_pel * h_PC * C_pel))) 
    du[3] = ((e_CR * a_CR_litt * R_litt *C_litt)/ (1 + a_CR_litt * h_CR * R_litt)) - ((a_PC_litt * C_litt * P)/ (1 + (a_PR_litt * h_PR * R_litt) + (a_PR_pel * h_PR  * R_pel) + (a_PC_litt * h_PC * C_litt) + (a_PC_pel * h_PC * C_pel))) - (m_C * C_litt)
    du[4] = ((e_CR * a_CR_pel * R_pel *C_pel)/ (1 + a_CR_pel * h_CR * R_pel)) - ((a_PC_pel * C_pel * P)/ (1 + (a_PR_litt * h_PR * R_litt) + (a_PR_pel * h_PR  * R_pel) + (a_PC_litt * h_PC * C_litt) + (a_PC_pel * h_PC * C_pel))) -  (m_C * C_pel)
    du[5] =(((e_PR * a_PR_litt * R_litt * P) + (e_PR * a_PR_pel * R_pel * P) + (e_PC * a_PC_litt * C_litt * P) + (e_PC * a_PC_pel * C_pel * P)) / (1 + (a_PR_litt * h_PR * R_litt) + (a_PR_pel * h_PR  * R_pel) + (a_PC_litt * h_PC * C_litt) + (a_PC_pel * h_PC * C_pel))) - (m_P * P)
    return du
end



let
    u0 = [ 0.6319695090,
    0.549944401,
    0.21748785,
    0.2053858,
    0.1619494]
    t_span = (0.0, 1000.0)
    p = AdaptPar(T=29)

    prob_adapt = ODEProblem(adapt_model!, u0, t_span, p)
    sol = OrdinaryDiffEq.solve(prob_adapt)

    adapt_ts = figure()
    plot(sol.t, sol.u)
    xlabel("time")
    ylabel("Density")
    legend(["R_litt", "R_pel", "C_litt", "C_pel", "P"])
    return adapt_ts

end

## Calculating Interior Eqs -- numerical solutions

tspan = (0.0, 1000.0)
u0 = [1.0, 1.0, 0.5, 0.5, 0.4]

par = AdaptPar(T=29)

prob = ODEProblem(adapt_model!, u0, tspan, par)

sol = OrdinaryDiffEq.solve(prob)

eq = nlsolve((du, u) -> adapt_model!(du, u, par, 0.0), sol.u[end]).zero

## output is numerical solutions to substitute into equations


## Adding stochasticity to model using gaussian white noise (SDEproblem)

function stoch_adapt!(du, u, p2, t)
    @unpack  noise = p2

    du[1] = noise * u[1]
    du[2] = noise * u[2]
    du[3] = noise * u[3]
    du[4] = noise * u[4]
    du[5] = noise * u[5]
    return du 
end


## Plotting time series with noise 

let
    param = AdaptPar(T=26, noise = 0.01)
    u0 = [0.5, 0.5, 0.5, 0.5, 0.5]
    tspan = (0.0, 1000.0)

    prob_stoch = SDEProblem(adapt_model!, stoch_adapt!, u0, tspan, param)
    sol_stoch = solve(prob_stoch, reltol = 1e-15, callback = PositiveDomain())

    stoch_ts = figure()
    plot(sol_stoch.t, sol_stoch.u)
    xlabel("time")
    ylabel("Density")
    legend(["R_litt", "R_pel", "C_litt", "C_pel", "P"])
    return stoch_ts

end
