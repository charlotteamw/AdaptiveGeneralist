using Parameters
using LinearAlgebra
using ForwardDiff
using PyPlot
using DifferentialEquations
using NLsolve

pygui(true)

# parameters declare them so they can be passed, mutable means you can change one or more
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

## using ForwardDiff for eigenvalue analysis, need to reassign model for just u
function adapt_model(u, AdaptPar, t)
    du = similar(u)
    adapt_model!(du, u, AdaptPar, t)
    return du
end

## calculating the jacobian 
function jac(u, model, p)
    ForwardDiff.jacobian(u -> model(u, p, NaN), u)
end
 
λ_stability(M) = maximum(real.(eigvals(M)))

adapt_jac = jac(equil, adapt_model, par)

eigvals(adapt_jac)

λ_stability(adapt_jac)

function T_maxeig_data()
    Tvals = 27.45:0.01:31.3
    max_eig = zeros(length(Tvals))
    u0 = [ 0.5, 0.5, 0.3, 0.3, 0.3]
    tspan = (0.0, 1000.0)

   for (Ti, Tval) in enumerate(Tvals)
        p = AdaptPar(T = Tval)
        prob = ODEProblem(adapt_model!, u0, tspan, p)
        sol = OrdinaryDiffEq.solve(prob)
        equilibrium = nlsolve((du, u) -> adapt_model!(du, u, p, 0.0), sol.u[end]).zero
        adapt_jac = jac(equilibrium, adapt_model, p)
        max_eig[Ti] = λ_stability(adapt_jac)
    end
 return hcat(collect(Tvals), max_eig)
end



let
    data = T_maxeig_data()
    maxeigen_plot = figure()
    y=0
    plot(data[:,1], data[:,2], y, color = "black")
    hlines= ([0])
    ylabel("Re(λₘₐₓ)", fontsize = 14, fontweight=:bold)
    xlabel("Temperature", fontsize = 14, fontweight=:bold)
    ylim(-0.1,0.1)
    xlim(27.5, 31.1)
    return maxeigen_plot
end


## printing out all five eigenvalues (real and complex) to determine where across temp axis eigs are only real or have complex part 

let 
    u0 = [ 0.5, 0.5, 0.3, 0.3, 0.3]
    tspan = (0.0, 1000.0)
    p = AdaptPar(T = 27.6)
    prob = ODEProblem(adapt_model!, u0, tspan, p)
    sol = OrdinaryDiffEq.solve(prob)
    equilibrium = nlsolve((du, u) -> adapt_model!(du, u, p, 0.0), sol.u[end]).zero
    adapt_jac = jac(equilibrium, adapt_model, p)
    eigs_all= eigvals(adapt_jac)
    println(eigs_all)
end