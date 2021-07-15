using Parameters
using LinearAlgebra
using ForwardDiff
using PyPlot
using DifferentialEquations
using NLsolve


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



function cr_model!(du, u, p, t)
    @unpack r_litt, k_litt, e_CR, e_PC, e_PR, aT_litt, a_CR_litt, a_PR_litt, h_CR, h_PC, h_PR, m_C, m_P, T, Topt_litt, Tmax_litt, aT_litt, σ = p 
    
    
    R_l, C_l, = u
    
    du[1]= r_litt * R_l * (1 -  R_l/k_litt) - (a_CR_litt * R_l * C_l)/(1 + (a_CR_litt * h_CR * R_l))
    

    du[2] = (e_CR * a_CR_litt * R_l * C_l)/(1 + (a_CR_litt * h_CR * R_l))  - m_C * C_l
    
    

    return 
end

let
    u0 = [0.5, 0.5]
    t_span = (0, 2000.0)
    ts = range(0, 1000, length = 1000)
    p = Par(T=29.0)
    prob1 = ODEProblem(cr_model!, u0, t_span, p)
    sol1 = solve(prob1, reltol = 1e-15)
    grid = sol1(ts)
    adapt_ts = figure()
    plot(grid.t, grid.u)
    xlabel("Time", fontsize=14,fontweight=:bold)
    ylabel("Density", fontsize=14,fontweight=:bold)
    legend(["R", "C"])
    return adapt_ts

end