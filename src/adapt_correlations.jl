using Parameters
using LinearAlgebra
using ForwardDiff
using PyPlot
using DifferentialEquations
using NLsolve
using StatsBase
using Distributed


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


## Calculating Cross Correlations 

let
    u0 = [0.8, 0.6, 0.5, 0.4, 0.3]
    t_span = (0, 1000.0)
    p = AdaptPar(T=27.5)

    prob_adapt = ODEProblem(adapt_model!, u0, t_span, p)
    sol = OrdinaryDiffEq.solve(prob_adapt, reltol = 1e-8, abstol = 1e-8)

    adapt_corr = crosscor(sol[1, 1:end], sol[2, 1:end], [0])
    return adapt_corr

end


function correlations(tend)
    Tvals = 27.5:0.01:31.0
    corr_temp = zeros(length(Tvals))
    u0 = [0.5, 0.5, 0.3, 0.3, 0.3]
    t_span = (0.0, 1000.0)
    time = 500.0:1.0:tend

    for (Ti, Tvals) in enumerate(Tvals)
        p = AdaptPar(T = Tvals)
        prob = ODEProblem(adapt_model!, u0, t_span, p)
        sol = DifferentialEquations.solve(prob, reltol = 1e-8)
        asol= sol(time)

        corr_temp[Ti] = crosscor(asol[1, 1:end], asol[2, 1:end], [0])
    end
    return hcat(Tvals, corr_temp)
end

let
    data = correlations(1000.0)
    corrplot = figure()
    plot(data[:,1], data[:,2])
    xlabel("Temp")
    ylabel("R1 R2 Cross Correlation")
    return corrplot
end

