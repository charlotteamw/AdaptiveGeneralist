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
using StatsBase

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
    u0 = [0.5, 0.5, 0.3, 0.3, 0.3]
    t_span = (0, 10000.0)
    p = AdaptPar(T=27, noise =0.05)
    ts = range(5000, 6000, length = 1000)

    prob_stoch = SDEProblem(adapt_model!, stoch_adapt!, u0, t_span, p)
    sol_stoch = solve(prob_stoch, reltol = 1e-15)
    grid_stoch = sol_stoch(ts)
    adapt_stochts = figure()
    plot(grid_stoch.t[1:end], grid_stoch[5,1:end])
    xlabel("Lag",fontsize=14,fontweight=:bold)
    ylabel("ACF",fontsize=14,fontweight=:bold)
    xlabel("Time")
    ylabel("Density")
    legend(["P"])
    return adapt_stochts

end



## Calculating autocorrelations for one temperature -- no loop yet 
let
    u0 = [0.5, 0.5, 0.3, 0.3, 0.3]
    tspan = (0.0, 10000.0)
    p = AdaptPar(T=27.5, noise=0.05)
    ts = range(5000, 6000, length = 1000)
    prob_stoch = SDEProblem(adapt_model!, stoch_adapt!, u0, tspan, p)
    sol_stoch = solve(prob_stoch, reltol = 1e-15)
    grid_stoch = sol_stoch(ts)
    
    plot_stochautocorr = figure()
    plot(autocor(grid_stoch[5, 1:end], 0:200))
    xlabel("Lag",fontsize=14,fontweight=:bold)
    ylabel("ACF",fontsize=14,fontweight=:bold)
    xlim(0,200)
    ylim(-1.0, 1.0)
    return plot_stochautocorr   

end

function local_max(ts)
    lmaxs = Float64[]
    for i in 2:(length(ts) - 3)
        if ts[i - 1] < ts[i] && ts[i] > ts[i + 1]
            push!(lmaxs, ts[i])
        end
    end
    return lmaxs
end


function acf_data()
    Tvals = 27.5:0.05:31.5
    acf_temp = zeros(length(Tvals))
    u0 = [ 0.5, 0.5, 0.3, 0.3, 0.3]
    tspan = (0.0, 6000.0)
    ts = range(5000, 6000, length = 1000)

   for (Ti, Tval) in enumerate(Tvals)
        p = AdaptPar(T = Tval)
        prob_stoch = SDEProblem(adapt_model!, stoch_adapt!, u0, tspan, p)
        sol_stoch = solve(prob_stoch, reltol = 1e-15)
        grid_sol= sol_stoch(ts)
        grid_sol.u
        grid_sol.t
        acf_temp[Ti] = autocor(grid_sol[5,1:end], 1:200)[1]
    end
 return hcat(collect(Tvals), acf_temp)
end


function max_acf_data()
    Tvals = 27.5:0.05:31.5
    max_acf = zeros(length(Tvals))
    u0 = [ 0.5, 0.5, 0.3, 0.3, 0.3]
    tspan = (0.0, 6000.0)
    ts = range(5000, 6000, length = 1000)

   for (Ti, Tval) in enumerate(Tvals)
        p = AdaptPar(T = Tval)

        max_acf[Ti] = maximum(local_max(acf_data()))
  
    end
 return hcat(collect(Tvals), max_acf)
end



data = max_acf_data()
println(acf_data())


let
    data = acf_data()
    acf_plot = figure()
    plot(data[:,1], data[:,2], color = "black")
    ylabel("Maximum ACF", fontsize = 14, fontweight=:bold)
    xlabel("Temperature", fontsize = 14, fontweight=:bold)
    ylim(0.9,1.0)
    xlim(27.5, 31.5)
    return acf_plot
end















