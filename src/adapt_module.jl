using Parameters
using LinearAlgebra
using ForwardDiff
using PyPlot
using DifferentialEquations
using NLsolve
using Statistics
using RecursiveArrayTools
using Noise
using SymPy

## Here we will illustrate the case of a generalist predator that ultimately has a climate-driven differential response to its prey. In this case, we change from 
# 20C-30C and show a generalist predator switching between prey in alternate habitats. Here, the generalist predator is omnivorous, and has different temperature responses in different habitats (littoral & pelagic)


## Parameters are categorized by macrohabitat -> parameters with "_litt" indicate littoral macrohabitat values and those with "_pel" indicate pelagic macrohabitat values  

## Parameter alitt in model is the temperature dependent attack rate of P on C_litt and apel is the temperature dependent attack rate of P on C_Pel. All other attack rates are not temp dependent

@with_kw mutable struct AdaptPar
    
    r_litt = 1.0
    r_pel = 1.0
    α_pel = 0.8   ##competitive influence of pelagic resource on littoral resource 
    α_litt = 0.8   ## competitve influence of littoral resource on pelagic resource
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
    a_CR_litt = 1.0
    a_CR_pel = 1.0
    a_PR_litt = 0.2 
    a_PR_pel = 0.2
    aT_litt = 1.0
    aT_pel = 1.0
    Tmax_litt = 35
    Topt_litt = 25
    Tmax_pel = 30
    Topt_pel = 24
    σ = 6
    T = 20
    noise = 0

    
end


## Omnivory Module with Temp Dependent Attack Rates (alitt => aPC in littoral zone; apel => aPC in pelagic zone)

function adapt_model!(du, u, p, t)
    @unpack r_litt, r_pel, k_litt, k_pel, α_pel, α_litt, e_CR, e_PC, e_PR, aT_pel, aT_litt, a_CR_litt, a_CR_pel, a_PR_litt, a_PR_pel, h_CR, h_PC, h_PR, m_C, m_P, T, Topt_litt, Tmax_litt, aT_litt, Topt_pel, Tmax_pel, aT_pel, σ = p 
    
    
    R_litt, R_pel, C_litt, C_pel, P = u
    
    du[1] = (r_litt * R_litt) * (1 - ((α_pel * R_litt + R_pel)/k_litt)) - ((a_CR_litt * R_litt * C_litt)/ (1+ a_CR_litt + h_CR + R_litt )) - ((a_PR_litt * R_litt * P)/ (1 + (a_PR_litt * h_PR * R_litt) + (a_PR_pel * h_PR  * R_pel) + (a_PC_litt * h_PC * C_litt) + (a_PC_pel * h_PC * C_pel))) 
    du[2] = (r_pel * R_pel) * (1 - ((α_litt * R_litt + R_pel)/k_pel)) - ((a_CR_pel * R_pel * C_pel)/ (1+ a_CR_pel + h_CR + R_pel )) - ((a_PR_pel * R_pel * P)/ (1 + (a_PR_litt * h_PR * R_litt) + (a_PR_pel * h_PR  * R_pel) + (a_PC_litt * h_PC * C_litt) + (a_PC_pel * h_PC * C_pel))) 
    du[3] = ((e_CR * a_CR_litt * R_litt *C_litt)/ (1 + a_CR_litt * h_CR * R_litt)) - ((a_PC_litt * C_litt * P)/ (1 + (a_PR_litt * h_PR * R_litt) + (a_PR_pel * h_PR  * R_pel) + (a_PC_litt * h_PC * C_litt) + (a_PC_pel * h_PC * C_pel))) - (m_C * C_litt)
    du[4] = ((e_CR * a_CR_pel * R_pel *C_pel)/ (1 + a_CR_pel * h_CR * R_pel)) - ((a_PC_pel * C_pel * P)/ (1 + (a_PR_litt * h_PR * R_litt) + (a_PR_pel * h_PR  * R_pel) + (a_PC_litt * h_PC * C_litt) + (a_PC_pel * h_PC * C_pel))) -  (m_C * C_pel)
    du[5] =(((e_PR * a_PR_litt * R_litt * P) + (e_PR * a_PR_pel * R_pel * P) + (e_PC * a_PC_litt * C_litt * P) + (e_PC * a_PC_pel * C_pel * P)) / (1 + (a_PR_litt * h_PR * R_litt) + (a_PR_pel * h_PR  * R_pel) + (a_PC_litt * h_PC * C_litt) + (a_PC_pel * h_PC * C_pel))) - (m_P * P)
    return 
end


function adapt_model(u, AdaptPar, t)
    du = similar(u)
    adapt_model!(du, u, AdaptPar, t)
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
    sol = OrdinaryDiffEq.solve(prob_adapt,  reltol = 1e-8, abstol = 1e-8, alg_hints =[:stiff])

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


#### Eigenvalues 

@vars R_litt R_pel C_litt C_pel P
@vars r_litt r_pel k_litt k_pel α_pel α_litt e_CR e_PC e_PR aT_pel aT_litt a_CR_litt a_CR_pel a_PR_litt a_PR_pel h_CR h_PC h_PR m_C m_P T Topt_litt Tmax_litt aT_litt Topt_pel Tmax_pel aT_pel σ 
#  need to make a symbolic parameter list, as `AdaptPar` is numeric
spar = Dict(
    :r_litt => r_litt,
    :r_pel => r_pel, 
    :k_litt => k_litt,
    :k_pel => k_pel,
    :α_litt => α_litt,
    :α_pel => α_pel,
    :h_CR => h_CR,
    :h_PC => h_PC, 
    :h_PR => h_PR,
    :e_CR => e_CR,
    :e_PC => e_PC, 
    :e_PR => e_PR,
    :m_C => m_C, 
    :m_P => m_P, 
    :a_CR_litt => a_CR_litt,
    :a_CR_pel => a_CR_pel,
    :a_PR_litt => a_PR_litt,
    :a_PR_pel => a_PR_pel,
    :aT_litt => aT_litt,
    :aT_pel => aT_pel,
    :Tmax_litt => Tmax_litt,
    :Topt_litt => Topt_litt,
    :Tmax_pel => Tmax_pel,
    :Topt_pel => Topt_pel,
    :σ => σ,
    :T => T
    )



f1, f2, f3, f4, f5 = adapt_model([R_litt, R_pel, C_litt, C_pel, P], spar, NaN)
#-
SymPy.solve(f1, R_litt)
#-
SymPy.solve(f1, C_litt)
#-
SymPy.solve(f1, R_pel)
#-
SymPy.solve(f1, C_pel)
#-
SymPy.solve(f1, P)
#-




# Utilities for doing eigenvalue analysis using autodiff
function rhs(u, AdaptPar)
    du = similar(u)
    adapt_model(du, u, AdaptPar, zero(u))
    return du
end

find_eq(u, AdaptPar) = nlsolve((du, u) -> adapt_model!(du, u, AdaptPar, zero(u)), u).zero
cmat(u, AdaptPar) = ForwardDiff.jacobian(x -> rhs(x, AdaptPar), u)
"""M is the community matrix, that can be calculated with `cmat(u, p)`"""

λ1_stability(M) = maximum(real.(eigvals(M)))

adapt_inteq_jac = jac(find_eq(AdaptPar()), adapt_model, AdaptPar())

eigvals(adapt_inteq_jac)

λ_stability(adapt_inteq_jac)

adapt_00_jac = jac([0.0,0.0,0.0,0.0,0.0], adapt_model, AdaptPar())

λ_stability(adapt_00_jac)

eigvals(adapt_00_jac)
