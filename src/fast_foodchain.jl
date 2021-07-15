using Parameters
using LinearAlgebra
using ForwardDiff
using PyPlot
using DifferentialEquations
using NLsolve
using StatsBase
using Distributed


@with_kw mutable struct Param 
    
    r_litt = 2.0
    k_litt = 2.0 
    h_CR = 0.3
    h_PC = 0.3
    h_PR = 0.3
    e_CR = 0.7
    e_PC = 0.7
    e_PR = 0.7
    m_C = 0.4
    m_P = 0.2
    a_CR_litt = 4.5
    a_PC_litt = 2.0
    a_PR_litt = 0.1
    noise = 0.1
    
end



function chain_model!(du, u, p, t)
    @unpack r_litt, k_litt, e_CR, e_PC, e_PR, a_CR_litt, a_PR_litt, a_PC_litt, h_CR, h_PC, h_PR, m_C, m_P = p 
    
    R_l, C_l, P = u
    
    du[1]= r_litt * R_l * (1 -  R_l/k_litt) - (a_CR_litt * R_l * C_l)/(1 + (a_CR_litt * h_CR * R_l)) - (a_PR_litt * R_l * P)/(1 + (a_PR_litt * h_PR * R_l  + a_PC_litt * h_PC * C_l))
    

    du[2] = (e_CR * a_CR_litt * R_l * C_l)/(1 + (a_CR_litt * h_CR * R_l)) - (a_PC_litt * C_l * P)/(1 + (a_PR_litt * h_PR * R_l  + a_PC_litt * h_PC * C_l) ) - m_C * C_l
    
    
    du[3] = (e_PR * a_PR_litt * R_l * P + e_PC * a_PC_litt * C_l * P )/(1 + (a_PR_litt * h_PR * R_l +  a_PC_litt * h_PC * C_l )) - m_P * P

    return 
end

## plotting ts for the fast chain 

let
    u0 = [0.5, 0.5, 0.5]
    t_span = (0, 1000.0)
    ts = range(0.0, 1000.0, length = 10000) 
    p = Param()
    prob = ODEProblem(chain_model!, u0, t_span, p)
    sol = solve(prob, reltol = 1e-15)
    grid_sol = sol(ts)
    chain_ts = figure()
    plot(grid_sol.t, grid_sol.u)
    xlabel("Time", fontsize=14,fontweight=:bold)
    ylabel("Density", fontsize=14,fontweight=:bold)
    legend(["R", "C","P"])
    return chain_ts

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