using Parameters
using LinearAlgebra
using ForwardDiff
using PyPlot
using DifferentialEquations
using NLsolve
using Statistics
using RecursiveArrayTools
using Noise
using Distributed

## Here we will illustrate the case of a generalist predator that ultimately has a climate-driven differential response to its prey. In this case, we change from 
# 20C-30C and show a generalist predator switching between prey in alternate habitats. Here, the generalist predator is omnivorous, and has different temperature responses in different habitats (littoral & pelagic)


## Parameters are categorized by macrohabitat -> parameters with "_litt" indicate littoral macrohabitat values and those with "_pel" indicate pelagic macrohabitat values  



@with_kw mutable struct AdaptPar 
    
    r_litt = 1.0
    r_pel = 1.0
    α_pel = 0.5   ##competitive influence of pelagic resource on littoral resource 
    α_litt = 0.5   ## competitve influence of littoral resource on pelagic resource
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
    a_CR_litt = 0.6
    a_CR_pel = 0.6
    a_PR_litt = 0.2 
    a_PR_pel = 0.2
    aT_litt = 1.5
    aT_pel = 1.5
    Tmax_litt = 35
    Topt_litt = 25
    Tmax_pel = 30
    Topt_pel = 24
    σ = 6
    T = 29
    noise = 0.1

    
end


## Omnivory Module with Temp Dependent Attack Rates (a_PC_litt => aPC in littoral zone; a_PC_pel => aPC in pelagic zone)

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
    
    R_l, R_p, C_l, C_p, P = u
    
    du[1]= r_litt * R_l * (1 - (α_pel * R_p + R_l)/k_litt) - (a_CR_litt * R_l * C_l)/(1 + a_CR_litt * h_CR * R_l) - (a_PR_litt * R_l * P)/(1 + a_PR_litt * h_PR * R_l + a_PR_pel * h_PR * R_p + a_PC_litt * h_PC * C_l + a_PC_pel * h_PC * C_p)
    
    du[2] = r_pel * R_p * (1 - (α_litt * R_l + R_p)/k_pel) - (a_CR_pel * R_p * C_p)/(1 + a_CR_pel * h_CR * R_p) - (a_PR_pel * R_p * P)/(1 + a_PR_litt * h_PR * R_l + a_PR_pel * h_PR * R_p + a_PC_litt * h_PC * C_l + a_PC_pel * h_PC * C_p)

    du[3] = (e_CR * a_CR_litt * R_l * C_l)/(1 + a_CR_litt * h_CR * R_l) - (a_PC_litt * C_l * P)/(1 + a_PR_litt * h_PR * R_l + a_PR_pel * h_PR * R_p + a_PC_litt * h_PC * C_l + a_PC_pel * h_PC * C_p) - m_C * C_l
    
    du[4] = (e_CR * a_CR_pel * R_p * C_p)/(1 + a_PC_pel * h_PC * R_p) - (a_PC_pel * C_p * P)/(1 + a_PR_litt * h_PR * R_l + a_PR_pel * h_PR * R_p + a_PC_litt * h_PC * C_l + a_PC_pel * h_PC * C_p) - m_C * C_p
    
    du[5] = (e_PR * a_PR_litt * R_l * P + e_PR * a_PR_pel * R_p * P + e_PC * a_PC_litt * C_l * P + e_PC * a_PC_pel * C_p * P)/(1 + a_PR_litt * h_PR * R_l + a_PR_pel * h_PR * R_p + a_PC_litt * h_PC * C_l + a_PC_pel * h_PC * C_p) - m_P * P

    return 
end


function adapt_model(u, AdaptPar, t)
    du = similar(u)
    adapt_model!(du, u, AdaptPar, t)
    return du
end

let
    u0 = [0.6802990085400237,
    0.5275129023434301,
    0.08959168230963603,
    0.23365213014702507,
    0.06954368400104854]
    t_span = (0.0, 1000.0)
    p = AdaptPar()

    prob_adapt = ODEProblem(adapt_model!, u0, t_span, p)
    sol = OrdinaryDiffEq.solve(prob_adapt,  reltol = 1e-8, abstol = 1e-8)

    adapt_ts = figure()
    plot(sol.t, sol.u)
    xlabel("time")
    ylabel("Density")
    legend(["R_litt", "R_pel", "C_litt", "C_pel", "P"])
    return adapt_ts

end

## Calculating Interior Eqs -- numerical solutions

tspan = (0.0, 1000.0)

u0 = [ 0.5, 0.5, 0.3, 0.3, 0.3]

par = AdaptPar(T=29)

prob = ODEProblem(adapt_model!, u0, tspan, par)

sol = OrdinaryDiffEq.solve(prob)

eq = nlsolve((du, u) -> adapt_model!(du, u, par, 0.0), sol.u[end]).zero

## output is numerical solutions to substitute into equations



## Stability Analysis

# Utilities for doing eigenvalue analysis using autodiff
function rhs(u, AdaptPar)
    du = similar(u)
    adapt_model!(du, u, AdaptPar, zero(u))
    return du
end

find_eq(u, AdaptPar) = nlsolve((du, u) -> adapt_model!(du, u, AdaptPar, zero(u)), u).zero

cmat(u, AdaptPar) = ForwardDiff.jacobian(x -> rhs(x, AdaptPar), u)

λ1_stability(M) = maximum(real.(eigvals(M)))



# Check mark diagram
function temp_maxeigen_data()
    tspan = (0.0, 1000.0)
    u0 = [0.5, 0.5, 0.3, 0.3, 0.3]
    p = AdaptPar(T=29)
    prob = ODEProblem(adapt_model!, u0, tspan, p)
    sol = OrdinaryDiffEq.solve(prob)
    Tvals = 25:0.0001:35
    max_eig = zeros(length(Tvals))

    for (ei, Tval) in enumerate(Tvals)
        eq_adapt = find_eq(sol[end], p)
        p = AdaptPar(T = Tval)
        max_eig[ei] = λ1_stability(cmat(eq_adapt, p))
    end
    return hcat(collect(Tvals), max_eig)
end

let
    data = temp_maxeigen_data()
    maxeigen_plot = figure()
    plot(data[:,1], data[:,2], color = "black")
    ylabel("Re(λₘₐₓ)", fontsize = 15)
    xlim(25, 35)
    ylim(-1.0, 1.0)
    xlabel("Temperature (C)", fontsize = 15)
    return maxeigen_plot
end



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
    param = AdaptPar(T=29, noise = 0.01)
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





