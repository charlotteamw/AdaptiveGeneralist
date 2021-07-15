using Parameters
using LinearAlgebra
using ForwardDiff
using PyPlot
using DifferentialEquations
using NLsolve
using StatsBase
using Distributed


@with_kw mutable struct Par 
    
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
    aT_litt = 2.0
    aT_pel = 2.0
    Tmax_litt = 35
    Topt_litt = 25
    Tmax_pel = 30
    Topt_pel = 24
    σ = 6
    T = 29
    noise = 0.1
    
end



function adapt_model!(du, u, p, t)
    @unpack r_litt, k_litt, e_CR, e_PC, e_PR, aT_litt, a_CR_litt, a_PR_litt, h_CR, h_PC, h_PR, m_C, m_P, T, Topt_litt, Tmax_litt, aT_litt, σ = p 
    
        
    a_PC_litt = ifelse(T < Topt_litt,  
    aT_litt * exp(-((T - Topt_litt)/(2*σ))^2), 
    aT_litt * (1 - ((T - (Topt_litt))/((Topt_litt) - Tmax_litt))^2)
    )
    
    
    R_l, C_l, P = u
    
    du[1]= r_litt * R_l * (1 -  R_l/k_litt) - (a_CR_litt * R_l * C_l)/(1 + (a_CR_litt * h_CR * R_l)) - (a_PR_litt * R_l * P)/(1 + (a_PR_litt * h_PR * R_l  + a_PC_litt * h_PC * C_l))
    

    du[2] = (e_CR * a_CR_litt * R_l * C_l)/(1 + (a_CR_litt * h_CR * R_l)) - (a_PC_litt * C_l * P)/(1 + (a_PR_litt * h_PR * R_l  + a_PC_litt * h_PC * C_l) ) - m_C * C_l
    
    
    du[3] = (e_PR * a_PR_litt * R_l * P + e_PC * a_PC_litt * C_l * P )/(1 + (a_PR_litt * h_PR * R_l +  a_PC_litt * h_PC * C_l )) - m_P * P

    return 
end

## plotting ts for the fast chain 

let
    u0 = [0.5, 0.5, 0.5]
    t_span = (0, 10000.0)
    ts = range(0.0, 10000.0, length = 10000) 
    p = Par(T=29.0, aT_litt = 8.0, m_C = 0.3, m_P = 0.3, k_litt = 1.0)
    prob = ODEProblem(adapt_model!, u0, t_span, p)
    sol = solve(prob, reltol = 1e-15)
    grid_sol = sol(ts)
    adapt_ts = figure()
    plot(grid_sol.t, grid_sol.u)
    xlabel("Time", fontsize=14,fontweight=:bold)
    ylabel("Density", fontsize=14,fontweight=:bold)
    legend(["R", "C","P"])
    return adapt_ts

end



## plotting ts for the slow chain 

let
    u0 = [0.5, 0.5, 0.5]
    t_span = (0, 2000.0)
    ts = range(0.0, 2000.0, length = 2000) 
    p = Par(T=29.0, aT_litt = 0.15, m_C = 0.15, m_P = 0.15, k_litt = 1.0)
    prob = ODEProblem(adapt_model!, u0, t_span, p)
    sol = solve(prob, reltol = 1e-15)
    grid_sol = sol(ts)
    adapt_ts = figure()
    plot(grid_sol.t, grid_sol.u)
    xlabel("Time", fontsize=14,fontweight=:bold)
    ylabel("Density", fontsize=14,fontweight=:bold)
    legend(["R", "C","P"])
    return adapt_ts

end