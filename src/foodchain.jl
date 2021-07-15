using Parameters
using LinearAlgebra
using ForwardDiff
using PyPlot
using DifferentialEquations
using NLsolve
using StatsBase
using Distributed


@with_kw mutable struct Pars 
    
    r = 1.0
    k = 1.0
    h_CR = 0.3
    h_PC = 0.3
    h_PR = 0.6
    e_CR = 0.8
    e_PC = 0.8
    e_PR = 0.6
    m_C = 0.2
    m_P = 0.3
    a_CR = 2.0
    aT_PC = 2.0
    aT_PR = 0.7
    Tmax_PC = 28
    Topt_PC = 20
    Tmax_PR = 35
    Topt_PR = 26
    σ = 6
    T = 25
    noise = 0.1
    
end



function fc_module!(du, u, p, t)
    @unpack r, k, e_CR, e_PC, e_PR, aT_PC, a_CR, h_CR, h_PC, h_PR, m_C, m_P, T, Topt_PC, Tmax_PC, aT_PC, Topt_PR, Tmax_PR, aT_PR, σ = p 
    
        
    a_PC= ifelse(T < Topt_PC,  
    aT_PC * exp(-((T - Topt_PC)/(2*σ))^2), 
    aT_PC * (1 - ((T - (Topt_PC))/((Topt_PC) - Tmax_PC))^2)
    )
    
    a_PR = ifelse(T < Topt_PR,  
    aT_PR * exp(-((T - Topt_PR)/(2*σ))^2), 
    aT_PR * (1 - ((T - (Topt_PR))/((Topt_PR) - Tmax_PR))^2)
    )
    
    
    R_l, C_l, P = u
    
    du[1]= r * R_l * (1 -  R_l/k) - (a_CR * R_l * C_l)/(1 + (a_CR * h_CR * R_l)) - (a_PR * R_l * P)/(1 + (a_PR * h_PR * R_l  + a_PC * h_PC * C_l))
    

    du[2] = (e_CR * a_CR * R_l * C_l)/(1 + (a_CR * h_CR * R_l)) - (a_PC * C_l * P)/(1 + (a_PR * h_PR * R_l  + a_PC * h_PC * C_l) ) - m_C * C_l
    
    
    du[3] = (e_PR * a_PR * R_l * P + e_PC * a_PC * C_l * P )/(1 + (a_PR * h_PR * R_l +  a_PC * h_PC * C_l )) - m_P * P

    return 
end

## plotting attack rates (temperature dependence)


###Time Series 
let
    u0 = [0.5, 0.5, 0.5]
    t_span = (0, 2000.0)
    ts = range(0.0, 2000.0, length = 2000) 
    p = Par(T=27.0) 
    prob = ODEProblem(fc_module!, u0, t_span, p)
    sol = solve(prob, reltol = 1e-15)
    grid_sol = sol(ts)
    adapt_ts = figure()
    plot(grid_sol.t, grid_sol.u)
    xlabel("Time", fontsize=14,fontweight=:bold)
    ylabel("Density", fontsize=14,fontweight=:bold)
    legend(["R", "C","P"])
    return adapt_ts

end

function a_data()
    Tvals = 10.0:0.1:35.0
    aPC_T = zeros(length(Tvals))
    aPR_T = zeros(length(Tvals))
    aT_PC = 1.5
    aT_PR = 0.7
    Tmax_PC = 30
    Topt_PC = 20
    Tmax_PR = 35
    Topt_PR = 26
    σ = 6
   for (Ti, Tval) in enumerate(Tvals)
       
        aPC_T[Ti] = ifelse(Tval < Topt_PC,  
        aT_PC * exp(-((Tval - Topt_PC)/(2*σ))^2), 
        aT_PC * (1 - ((Tval - (Topt_PC))/((Topt_PC) - Tmax_PC))^2)
        )

        aPR_T[Ti] = ifelse(Tval < Topt_PR,  
        aT_PR * exp(-((Tval - Topt_PR)/(2*σ))^2), 
        aT_PR * (1 - ((Tval - (Topt_PR))/((Topt_PR) - Tmax_PR))^2)
        )
    end
 return hcat(collect(Tvals), aPC_T, aPR_T)
end


let
    data = a_data()
    aT_plot = figure()
    plot(data[:,1], data[:,2],  color = "orange")
    plot(data[:,1], data[:,3], color = "green")
    ylabel("Attack Rate", fontsize = 14, fontweight=:bold)
    xlabel("Temperature", fontsize = 14, fontweight=:bold)
    ylim(0, 3.0)
    xlim(10.0, 35.0)
    legend(["aPC", "aPR"])
    return aT_plot
end




function omnivory_data()
    Tvals = 10.0:0.01:30.0
    u0 = [0.5,0.5,0.5]
    omnivory = zeros(length(Tvals))
   
    r = 1.0
    k = 1.0
    h_CR = 0.3
    h_PC = 0.3
    h_PR = 0.6
    e_CR = 0.8
    e_PC = 0.8
    e_PR = 0.6
    m_C = 0.2
    m_P = 0.3
    a_CR = 2.0
    aT_PC = 2.0
    aT_PR = 0.7
    Tmax_PC = 28
    Topt_PC = 20
    Tmax_PR = 35
    Topt_PR = 26
    σ = 6
    T = 25
    noise = 0.1
    
   for (Ti, Tval) in enumerate(Tvals)
        p= Par(T=Tval)
       
        a_PC = ifelse(Tval < Topt_PC,  
        aT_PC * exp(-((Tval - Topt_PC)/(2*σ))^2), 
        aT_PC * (1 - ((Tval- (Topt_PC))/((Topt_PC) - Tmax_PC))^2)[1]
        )

        a_PR = ifelse(Tval < Topt_PR,  
        aT_PR * exp(-((Tval - Topt_PR)/(2*σ))^2), 
        aT_PR * (1 - ((Tval - (Topt_PR))/((Topt_PR) - Tmax_PR))^2)
        )

        eq= (nlsolve((du, u) -> fc_module!(du, u, p, 0.0), u0).zero)
        omnivory[Ti] = ((a_PR * e_PR * eq[3] * eq[1])/(1 + h_PR * a_PR * eq[1]))/(((a_PR * e_PR * eq[3] * eq[1])/(1 + h_PR * a_PR * eq[1])) + ((a_PC * e_PC * eq[3] * eq[2])/(1 + h_PC * a_PC * eq[2])))

    end
 return hcat(collect(Tvals), omnivory)
end

println(omnivory_data())
let
    data = omnivory_data()
    omnivory_plot = figure()
    plot(data[:,1], data[:,2])
    ylabel("Omnivory", fontsize = 14, fontweight=:bold)
    xlabel("Temperature", fontsize = 14, fontweight=:bold)
    ylim(0, 1.0)
    xlim(10.0, 26.5)
    return omnivory_plot
end



## Stability Analysis

## using ForwardDiff for eigenvalue analysis, need to reassign model for just u
function fc_module(u, Par, t)
    du = similar(u)
    fc_module!(du, u, Par, t)
    return du
end

## calculating the jacobian 
function jac(u, model, p)
    ForwardDiff.jacobian(u -> model(u, p, NaN), u)
end
 
λ_stability(M) = maximum(real.(eigvals(M)))


function T_maxeig_data()
    Tvals = 0.0:0.01:30.0
    max_eig = zeros(length(Tvals))
    u0 = [ 0.5, 0.5, 0.5]
    tspan = (0.0, 2000.0)

   for (Ti, Tval) in enumerate(Tvals)
        p = Pars(T = Tval)
        prob = ODEProblem(fc_module!, u0, tspan, p)
        sol = OrdinaryDiffEq.solve(prob)
        equilibrium = nlsolve((du, u) -> fc_module!(du, u, p, 0.0), sol.u[end]).zero
        fc_jac = jac(equilibrium, fc_module, p)
        max_eig[Ti] = λ_stability(fc_jac)
    end
 return hcat(collect(Tvals), max_eig)
end



let
    data = T_maxeig_data()
    maxeigen_plot = figure()
    plot(data[:,1], data[:,2], color = "black")
    hlines= ([0])
    ylabel("Re(λₘₐₓ)", fontsize = 14, fontweight=:bold)
    xlabel("Temperature", fontsize = 14, fontweight=:bold)
    ylim(-0.2,0.1)
    xlim(9.185, 26.24)
    return maxeigen_plot
end



let 
    u0 = [ 0.5, 0.5, 0.5]
    tspan = (0.0, 2000.0)
    p = Par(T = 26.245)
    prob = ODEProblem(fc_module!, u0, tspan, p)
    sol = OrdinaryDiffEq.solve(prob)
    equilibrium = nlsolve((du, u) -> fc_module!(du, u, p, 0.0), sol.u[end]).zero
    fc_jac = jac(equilibrium, fc_module, p)
    eigs_all= eigvals(fc_jac)
    println(eigs_all)
end