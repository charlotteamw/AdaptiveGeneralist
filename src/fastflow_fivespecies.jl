using Parameters
using LinearAlgebra
using ForwardDiff
using PyPlot
using DifferentialEquations
using NLsolve


## Here we will illustrate the case of a generalist predator that ultimately has a climate-driven differential response to its prey. In this case, we change from 
# 20C-30C and show a generalist predator switching between prey in alternate habitats. Here, the generalist predator is omnivorous, and has different temperature responses in different habitats (littoral & pelagic)


## Parameters are categorized by macrohabitat -> parameters with "_litt" indicate littoral macrohabitat values and those with "_pel" indicate pelagic macrohabitat values  



@with_kw mutable struct GenPar 
    
    r_litt = 1.0
    r_pel = 1.0
    α_pel = 0.5   ##competitive influence of pelagic resource on littoral resource 
    α_litt = 0.5   ## competitve influence of littoral resource on pelagic resource
    h_CR_litt = 0.3
    h_CR_pel = 0.7
    h_PC_litt = 0.5
    h_PC_pel = 0.3
    h_PR_litt = 0.3
    h_PR_pel = 0.2
    e_CR_litt = 0.5
    e_CR_pel = 0.5
    e_PC_litt = 0.5
    e_PC_pel = 0.5
    e_PR_litt = 0.5
    e_PR_pel = 0.5
    m_C_litt = 0.4
    m_C_pel = 0.2
    m_P = 0.3
    a_CR_litt = 3.0 
    a_CR_pel = 1.0
    a_PR_litt = 0.1 
    a_PR_pel = 0.03
    a_PC_litt = 3.0 
    a_PC_pel = 0.2
    kfast = 1.0
    kslow = 0.4
    kfin = 0.5
    ksin = 0.5
    n = 1.0 

end
## Omnivory Module with Temp Dependent Attack Rates (a_PC_litt => aPC in littoral zone; a_PC_pel => aPC in pelagic zone)

function generalist_model!(du, u, p, t)
    @unpack r_litt, r_pel, kfast, kslow, kfin, ksin, n, α_pel, α_litt, e_CR_litt, e_CR_pel, e_PC_litt, e_PC_pel, e_PR_litt, e_PR_pel, a_CR_litt, a_CR_pel, a_PR_litt, a_PR_pel, a_PC_litt, a_PC_pel, h_CR_litt, h_CR_pel, h_PC_litt, h_PC_pel, h_PR_litt, h_PR_pel, m_C_litt, m_C_pel, m_P = p 
    
    R_l, R_p, C_l, C_p, P = u
    
    du[1]= r_litt * R_l * (1 - (α_pel * R_p + R_l)/k_litt()) - (a_CR_litt * R_l * C_l)/(1 + a_CR_litt * h_CR_litt * R_l) - (a_PR_litt * R_l * P)/(1 + a_PR_litt * h_PR_litt * R_l + a_PR_pel * h_PR_pel * R_p + a_PC_litt * h_PC_litt * C_l + a_PC_pel * h_PC_pel * C_p)
    
    du[2] = r_pel * R_p * (1 - (α_litt * R_l + R_p)/k_pel()) - (a_CR_pel * R_p * C_p)/(1 + a_CR_pel * h_CR_pel * R_p) - (a_PR_pel * R_p * P)/(1 + a_PR_litt * h_PR_litt * R_l + a_PR_pel * h_PR_pel * R_p + a_PC_litt * h_PC_litt * C_l + a_PC_pel * h_PC_pel * C_p)

    du[3] = (e_CR_litt * a_CR_litt * R_l * C_l)/(1 + a_CR_litt * h_CR_litt * R_l) - (a_PC_litt * C_l * P)/(1 + a_PR_litt * h_PR_litt * R_l + a_PR_pel * h_PR_pel * R_p + a_PC_litt * h_PC_litt * C_l + a_PC_pel * h_PC_pel * C_p) - m_C_litt * C_l
    
    du[4] = (e_CR_pel * a_CR_pel * R_p * C_p)/(1 + a_PC_pel * h_PC_pel * R_p) - (a_PC_pel * C_p * P)/(1 + a_PR_litt * h_PR_litt * R_l + a_PR_pel * h_PR_pel * R_p + a_PC_litt * h_PC_litt * C_l + a_PC_pel * h_PC_pel * C_p) - m_C_pel * C_p
    
    du[5] = (e_PR_litt * a_PR_litt * R_l * P + e_PR_pel * a_PR_pel * R_p * P + e_PC_litt * a_PC_litt * C_l * P + e_PC_pel * a_PC_pel * C_p * P)/(1 + a_PR_litt * h_PR_litt * R_l + a_PR_pel * h_PR_pel * R_p + a_PC_litt * h_PC_litt * C_l + a_PC_pel * h_PC_pel * C_p) - m_P * P

    return 
end


function k_litt(GenPar)
    kfast * n + kfin
end 

function k_pel(GenPar)
    kslow * n + ksin
end 

function RlittFR(GenPar)
    ((e_PR_litt * a_PR_litt * R_l * P)/(1 + a_PR_litt * h_PR_litt * R_l + a_PR_pel * h_PR_pel * R_p + a_PC_litt * h_PC_litt * C_l + a_PC_pel * h_PC_pel * C_p)) 
end

function RpelFR(GenPar)
    ((e_PR_pel * a_PR_pel * R_P * P)/(1 + a_PR_litt * h_PR_litt * R_l + a_PR_pel * h_PR_pel * R_p + a_PC_litt * h_PC_litt * C_l + a_PC_pel * h_PC_pel * C_p)) 
end

function ClittFR(GenPar)
    ((e_PC_litt * a_PC_litt * C_l * P)/(1 + a_PR_litt * h_PR_litt * R_l + a_PR_pel * h_PR_pel * R_p + a_PC_litt * h_PC_litt * C_l + a_PC_pel * h_PC_pel * C_p)) 
end

function CpelFR(GenPar)
    ((e_PC_pel * a_PC_pel * C_p * P)/(1 + a_PR_litt * h_PR_litt * R_l + a_PR_pel * h_PR_pel * R_p + a_PC_litt * h_PC_litt * C_l + a_PC_pel * h_PC_pel * C_p)) 
end 

























## Plotting time series - deterministic model 





let
    u0 = [0.5, 0.5, 0.3, 0.3, 0.3]
    t_span = (0, 10000.0)
    p = GenPar(n = 2.0)
    ts = range(9000, 10000, length = 1000)
    prob = ODEProblem(generalist_model!, u0, t_span, p)
    sol = solve(prob, reltol = 1e-15)
    grid = sol(ts)
    gen_ts = figure()
    plot(grid.t, grid.u)
    xlabel("Time")
    ylabel("Density")
    legend(["R1", "R2", "C1", "C2", "P"])
    return gen_ts

end

## using ForwardDiff for eigenvalue analysis, need to reassign model for just u
function generalist_model(u, GenPar, t)
    du = similar(u)
    generalist_model!(du, u, GenPar, t)
    return du
end

## calculating the jacobian 
function jac(u, model, p)
    ForwardDiff.jacobian(u -> model(u, p, NaN), u)
end
 
λ_stability(M) = maximum(real.(eigvals(M)))


## calc equilibrium
p = GenPar(n = 2.2)
u0 = [0.5, 0.5, 0.3, 0.3, 0.3]
tspan = (0.0, 10000.0)
ts = range(9000, 10000, length = 1000)
prob = ODEProblem(generalist_model!, u0, tspan, p)
sol = OrdinaryDiffEq.solve(prob)
grid = sol(ts)
equilibrium = nlsolve((du, u) -> generalist_model!(du, u, p, 0.0), grid.u[end]).zero 

function n_maxeig_data()
    Nvals = 0.8:0.005:2.2
    max_eig = zeros(length(Nvals))
    u0 = [ 0.5, 0.5, 0.3, 0.3, 0.3]
    tspan = (0.0, 10000.0)
    ts = range(9000, 10000, length = 1000)
  

   for (Ni, Nval) in enumerate(Nvals)
        p = GenPar(n = Nval)
        u0 = [0.5, 0.5, 0.3, 0.3, 0.3]
        prob = ODEProblem(generalist_model!, u0, tspan, p)
        sol = OrdinaryDiffEq.solve(prob)
        grid = sol(ts)
        equilibrium = nlsolve((du, u) -> generalist_model!(du, u, p, 0.0), grid.u[end]).zero     
        gen_jac = jac(equilibrium, generalist_model, p)
        max_eig[Ni] = λ_stability(gen_jac)
    end
 return hcat(collect(Nvals), max_eig)
end

println(n_maxeig_data())

let
    data = n_maxeig_data()
    maxeigen_plot = figure()
    plot(data[:,1], data[:,2], color = "black")
    ylabel("Re(λₘₐₓ)", fontsize = 14, fontweight=:bold)
    xlabel("Nutrient Concentration", fontsize = 14, fontweight=:bold)
    ylim(-0.1,0.3)
    xlim(0.8, 2.2)
    return maxeigen_plot
end


let 
    u0 = [ 0.5, 0.5, 0.3, 0.3, 0.3]
    tspan = (0.0, 1000.0)
    ts = range(0, 2000, length = 2000)
    p = GenPar(n = 2.0)
    eq = nlsolve((du, u) -> generalist_model!(du, u, p, 0.0), u0).zero
    gen_jac = jac(eq, generalist_model, p)
    eigs_all= eigvals(gen_jac)
    println(eigs_all)
end



function n_couple_data()
    Nvals = 0.8:0.005:2.2
    u0 = [ 0.5, 0.5, 0.3, 0.3, 0.3]
    tspan = (0.0, 10000.0)
    ts = range(9000, 10000, length = 1000)
    coupling = zeros(length(Nvals))

    h_CR_litt = 0.3
    h_CR_pel = 0.7
    h_PC_litt = 0.5
    h_PC_pel = 0.3
    h_PR_litt = 0.3
    h_PR_pel = 0.2
    e_CR_litt = 0.5
    e_CR_pel = 0.5
    e_PC_litt = 0.5
    e_PC_pel = 0.5
    e_PR_litt = 0.5
    e_PR_pel = 0.5
    a_CR_litt = 3.0 
    a_CR_pel = 1.0
    a_PR_litt = 0.1 
    a_PR_pel = 0.03
    a_PC_litt = 3.0 
    a_PC_pel = 0.2
    kfast = 1.5
    kslow = 1.0
    kfin = 0.2
    ksin = 0.1

    for (Ni, Nval) in enumerate(Nvals)
        p = GenPar(n = Nval)
        prob = ODEProblem(generalist_model!, u0, tspan, p)
        sol = OrdinaryDiffEq.solve(prob)
        grid = sol(ts)
        eq= nlsolve((du, u) -> generalist_model!(du, u, p, 0.0), grid.u[end]).zero  
        k_litt = kfast * Nval + kfin
        k_pel = kslow * Nval + ksin
        coupling[Ni] = ((e_PR_litt * a_PR_litt * R_l * P)/(1 + a_PR_litt * h_PR_litt * R_l + a_PR_pel * h_PR_pel * R_p + a_PC_litt * h_PC_litt * C_l + a_PC_pel * h_PC_pel * C_p)) + ((e_PC_litt * a_PC_litt * C_l * P)/(1 + a_PR_litt * h_PR_litt * R_l + a_PR_pel * h_PR_pel * R_p + a_PC_litt * h_PC_litt * C_l + a_PC_pel * h_PC_pel * C_p))
    end
    return hcat(collect(Nvals), coupling)
end




let
    data = n_couple_data()
    coupling_plot = figure()
    plot(data[:,1], data[:,2], color = "black")
    ylabel("% Fast Chain Carbon", fontsize = 15)
    xlim(0.8, 2.2)
    ylim(0.0, 1.0)
    xlabel("Nutrient Concentration", fontsize = 15)
    return coupling_plot
end













