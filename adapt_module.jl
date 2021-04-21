using Parameters
using LinearAlgebra
using ForwardDiff
using PyPlot
using DifferentialEquations
using NLsolve


## Here, we will illustrate the case of a generalist predator that ultimately has a climate-driven differential response to its prey. In this case, we change from 

## 20C-30C and show a generalist predator switching between prey in alternate habitats. Here, the generalist predator is omnivorous, and has different temperature responses in different habitats (1&2)

function a_PC_litt(u, p, t)
    @unpack T, Topt_litt, Tmax_litt, aT_litt, σ = p 
    R_litt, R_pel, C_litt, C_pel, P = u
    if T < Topt_litt
        return aT_litt * exp(-((T - Topt_litt)/(2 \σ))^2)
      else T >= Topt_litt
          return aT_litt * (1 - ((T - (Topt_litt))/((Topt_litt) - Tmax_litt))^2)
      end  
end

function a_PC_pel(u, p, t)
    @unpack T, Topt_pel, Tmax_pel, aT_pel, σ = p
    R_litt, R_pel, C_litt, C_pel, P = u
    if T < Topt_pel
        return aT_pel * exp(-((T - Topt_pel)/(2 \σ))^2)
      else T >= Topt_pel
          return aT_pel * (1 - ((T - (Topt_pel))/((Topt_pel) - Tmax_pel))^2)   
      end   
end

@with_kw mutable struct AdaptPar
    r_litt = 1.0
    k_litt = 1.0
    α_pel = 0.8      ##competitive influence of pelagic resource on littoral resource 
    r_pel = 1.0
    α_litt = 0.8     ##competitive influence of littoral resource on pelagic resource 
    k_pel = 1.0
    e_CR = 0.8
    h_CR = 0.5
    m_C = 0.2
    a_CR_litt = 1.0
    a_CR_pel = 1.0
    h_PC = 0.5
    h_PR = 0.5
    e_PC = 0.8
    e_PR = 0.8
    m_P = 0.3
    a_PR_litt = 0.2 
    a_PR_pel = 0.2 
    aT_litt = 1.0
    aT_pel = 1.0
    Tmax_litt = 40
    Topt_litt = 32
    Tmax_pel = 32
    Topt_pel = 25
    σ= 6
    T=30
    
end


function adapt_model!(du, u, p, t,)
    @unpack r_litt, r_pel, k_litt, k_pel, α_pel, α_litt, e_CR, e_PC, e_PR, aT_pel, aT_litt, a_CR_litt, a_CR_pel, a_PR_litt, a_PR_pel, h_CR, h_PC, h_PR, m_C, m_P = p 
   
    alitt =  if T < Topt_pel
        return aT_pel * exp(-((T - Topt_pel)/(2 \σ))^2)
      else T >= Topt_pel
          return aT_pel * (1 - ((T - (Topt_pel))/((Topt_pel) - Tmax_pel))^2)   
      end   
    aepl = if T < Topt_litt
        return aT_litt * exp(-((T - Topt_litt)/(2 \σ))^2)
      else T >= Topt_litt
          return aT_litt * (1 - ((T - (Topt_litt))/((Topt_litt) - Tmax_litt))^2)
      end  

    alitt =  a_PC_litt(u, p, t)
    apel = a_PC_pel(u, p, t)

    R_litt, R_pel, C_litt, C_pel, P = u

    du[1] = r_litt * R_litt * (1 - (α_pel * R_pel + R_litt)/k_litt) - (a_CR_litt * R_litt * C_litt)/( 1 + a_CR_litt * h_CR * R_litt) - (a_PR_litt * R_litt * P)/(1 + a_PR_litt * h_PR * R_litt + a_PR_pel * h_PR * R_pel + alitt * h_PC * C_litt +  apel * h_PC * C_pel)
    du[2] = r_pel * R_pel * (1 - (α_litt * R_litt + R_pel)/k_pel) - (a_CR_pel * R_pel * C_pel)/(1 + a_CR_pel * h_CR * R_pel) - (a_CR_pel * R_pel * P)/(1 + a_PR_litt * h_PR * R_litt + a_PR_pel * h_PR * R_pel + alitt * h_PC * C_litt +  apel * h_PC * C_pel)
    du[3] = (e_CR * a_CR_litt * R_litt * C_litt)/(1 + a_CR_litt * h_CR * R_litt) - (alitt * C_litt * P)/(1 + a_PR_litt * h_PR * R_litt + a_PR_pel * h_PR * R_litt + alitt * h_PC * C_litt +  apel * h_PC * C_pel) - m_C * C_litt
    du[4] = (e_CR * a_CR_pel * R_pel * C_pel)/(1 + a_CR_pel * h_CR * R_pel) - (apel * C_pel * P)/(1 + alitt * h_PR * R_litt +  apel * h_PR * R_pel + alitt * h_PC * C_litt +  apel * h_PC * C_pel) - m_C * C_pel 
    du[5] = (e_PR * a_PR_litt * R_litt * P * e_PR * a_PR_pel * R_pel * P + e_PC * alitt * C_litt * P + e_PC *  apel * C_pel * P)/(1 + a_PR_litt * h_PR * R_litt + a_PR_pel * h_PR * R_pel + alitt * h_PC * C_litt +  apel * h_PC * C_pel) - m_P * P

    return 
end

function adapt_model(u, par)
    du = similar(u)
    adapt_model!(du, u, par, 0.0)
    return du
end

tspan = (0.0, 1000.0)
u0 = [2.0, 1.0, 0.5,0.5,0.5]

par = AdaptPar()
prob = ODEProblem(adapt_model!, u0, tspan, par)

sol = solve(prob)

plot(sol)

eq = nlsolve((du, u) -> adapt_model!(du, u, par, 0.0), sol.u[end]).zero

let
    nsamp = 100
    avals = range(0.4, 1.2, length = nsamp)
    stab = fill(NaN, 2, nsamp)

    for (i, a) in enumerate(avals)
        par.a = a
        prob = ODEProblem(adapt_model!, u0, tspan, par)
        sol = solve(prob)
        eq = nlsolve((du, u) -> adapt_model!(du, u, par, 0.0), sol.u[end]).zero
        stab[1, i] = calc_λ1(eq, par)
        # This is using the equilibrium approximation, but maybe it is worth numerically
        # calculating the reactivity in the excitable case to see if it acts more "cycle like"
        # That is do we see an early onset of the reactivity increasing if we look at the local
        # distribution of deviations around the complex eigenvalues
        stab[2, i] = calc_ν(eq, par)
    end

    figure()
    subplot(311)
    plot(avals, stab[1, :])
    xlabel("a")
    ylabel(L"\lambda_1")

    subplot(312)
    plot(avals, stab[2, :])
    xlabel("a")
    ylabel(L"\nu")

    subplot(313)
    plot(stab[1, :], stab[2, :], "o")
    xlabel(L"\lambda_1")
    ylabel(L"\nu")
    tight_layout()
end



















