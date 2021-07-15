using Parameters
using LinearAlgebra
using ForwardDiff
using PyPlot
using DifferentialEquations
using NLsolve
using Statistics
using StatsBase

@with_kw mutable struct Par
    
    r_litt = 1.0
    r_pel = 1.0
    α_pel = 0.5  
    α_litt = 0.5  
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

## generalist module with fast and slow pathways 
function model!(du, u, p, t)
    @unpack r_litt, r_pel, kfast, kslow, kfin, ksin, n, α_pel, α_litt, e_CR_litt, e_CR_pel, e_PC_litt, e_PC_pel, e_PR_litt, e_PR_pel, a_CR_litt, a_CR_pel, a_PR_litt, a_PR_pel, a_PC_litt, a_PC_pel, h_CR_litt, h_CR_pel, h_PC_litt, h_PC_pel, h_PR_litt, h_PR_pel, m_C_litt, m_C_pel, m_P = p 
    
    R_l, R_p, C_l, C_p, P = u

    k_litt = kfast * n + kfin 
    k_pel = kslow * n + ksin
    
    du[1]= r_litt * R_l * (1 - (α_pel * R_p + R_l)/ k_litt) - (a_CR_litt * R_l * C_l)/(1 + a_CR_litt * h_CR_litt * R_l) - (a_PR_litt * R_l * P)/(1 + a_PR_litt * h_PR_litt * R_l + a_PR_pel * h_PR_pel * R_p + a_PC_litt * h_PC_litt * C_l + a_PC_pel * h_PC_pel * C_p)
    
    du[2] = r_pel * R_p * (1 - (α_litt * R_l + R_p)/ k_pel) - (a_CR_pel * R_p * C_p)/(1 + a_CR_pel * h_CR_pel * R_p) - (a_PR_pel * R_p * P)/(1 + a_PR_litt * h_PR_litt * R_l + a_PR_pel * h_PR_pel * R_p + a_PC_litt * h_PC_litt * C_l + a_PC_pel * h_PC_pel * C_p)

    du[3] = (e_CR_litt * a_CR_litt * R_l * C_l)/(1 + a_CR_litt * h_CR_litt * R_l) - (a_PC_litt * C_l * P)/(1 + a_PR_litt * h_PR_litt * R_l + a_PR_pel * h_PR_pel * R_p + a_PC_litt * h_PC_litt * C_l + a_PC_pel * h_PC_pel * C_p) - m_C_litt * C_l
    
    du[4] = (e_CR_pel * a_CR_pel * R_p * C_p)/(1 + a_PC_pel * h_PC_pel * R_p) - (a_PC_pel * C_p * P)/(1 + a_PR_litt * h_PR_litt * R_l + a_PR_pel * h_PR_pel * R_p + a_PC_litt * h_PC_litt * C_l + a_PC_pel * h_PC_pel * C_p) - m_C_pel * C_p
    
    du[5] = (e_PR_litt * a_PR_litt * R_l * P + e_PR_pel * a_PR_pel * R_p * P + e_PC_litt * a_PC_litt * C_l * P + e_PC_pel * a_PC_pel * C_p * P)/(1 + a_PR_litt * h_PR_litt * R_l + a_PR_pel * h_PR_pel * R_p + a_PC_litt * h_PC_litt * C_l + a_PC_pel * h_PC_pel * C_p) - m_P * P

    return 
end

## time series 
let
    u0 = [0.5, 0.5, 0.3, 0.3, 0.3]
    t_span = (0, 10000.0)
    p = Par(n = 1.3)
    ts = range(0.0, 5000.0, length = 5000)
    prob = ODEProblem(model!, u0, t_span, p)
    sol = solve(prob, reltol = 1e-15)
    grid = sol(ts)
    generalist_ts = figure()
    plot(grid.t, grid.u)
    xlabel("Time")
    ylabel("Density")
    legend(["Rf", "Rs", "Cf", "Cs", "P"])
    return generalist_ts

end


## equilibrium check for parameter range -- find where all species coexist (interior equilibrium)

## looks like n = 0.19 to 1.66 

vals = 0.0:0.005:1.66
nhold = fill(0.0,length(Nvals),6)

for i=1:length(Nvals)
    p = Par(n=Nvals[i])
    u0 = [ 0.5, 0.5, 0.3, 0.3, 0.3]
    tspan = (0.0, 10000.0)
    ts = range(9000, 10000, length = 1000)
    u0 = [0.5, 0.5, 0.3, 0.3, 0.3]
    prob = ODEProblem(model!, u0, tspan, p)
    sol = OrdinaryDiffEq.solve(prob)
    grid = sol(ts)
    eq = nlsolve((du, u) -> model!(du, u, p, 0.0), grid.u[end]).zero 
    nhold[i,1] = Nvals[i]
    nhold[i,2:end] = eq
    println(nhold[i,:])
end

## Eigenvalue analysis 
## using ForwardDiff for eigenvalue analysis, need to reassign model for just u
function model(u, Par, t)
    du = similar(u)
    model!(du, u, Par, t)
    return du
end

## calculating the jacobian 
function jac(u, model, p)
    ForwardDiff.jacobian(u -> model(u, p, NaN), u)
end

## getting max real eigenvalue 
λ_stability(M) = maximum(real.(eigvals(M)))

function n_maxeig_data()
    Nvals = 0.185:0.005:1.66
    max_eig = zeros(length(Nvals))
    u0 = [ 0.5, 0.5, 0.3, 0.3, 0.3]
    tspan = (0.0, 10000.0)
    ts = range(9000, 10000, length = 1000)
  

   for (Ni, Nval) in enumerate(Nvals)
        p = Par(n = Nval)
        u0 = [0.5, 0.5, 0.3, 0.3, 0.3]
        prob = ODEProblem(model!, u0, tspan, p)
        sol = OrdinaryDiffEq.solve(prob)
        grid = sol(ts)
        equilibrium = nlsolve((du, u) -> model!(du, u, p, 0.0), grid.u[end]).zero     
        generalist_jac = jac(equilibrium, model, p)
        max_eig[Ni] = λ_stability(generalist_jac)
    end
 return hcat(collect(Nvals), max_eig)
end

let
    data = n_maxeig_data()
    maxeigen_plot = figure()
    plot(data[:,1], data[:,2], color = "black")
    hlines(0.0, 0.0, 3.0, colors="grey")
    ylabel("Re(λₘₐₓ)", fontsize = 14, fontweight=:bold)
    xlabel("Nutrient Concentration", fontsize = 14, fontweight=:bold)
    ylim(-0.05, 0.025)
    xlim(0.185, 1.66)
    return maxeigen_plot
end


##P:C ratios
function n_ratio_data()
    Nvals = 0.19:0.005:1.66
    eq =zeros(length(Nvals))
    PRlitt = zeros(length(Nvals))
    PRpel =  zeros(length(Nvals))
    PClitt =  zeros(length(Nvals))
    PCpel =  zeros(length(Nvals))
    u0 = [ 0.5, 0.5, 0.3, 0.3, 0.3]
    tspan = (0.0, 10000.0)
    ts = range(9000, 10000, length = 1000)
  

   for (Ni, Nval) in enumerate(Nvals)
        p = Par(n = Nval)
        u0 = [ 0.5, 0.5, 0.3, 0.3, 0.3]
        tspan = (0.0, 10000.0)
        ts = range(9000, 10000, length = 1000)
        u0 = [0.5, 0.5, 0.3, 0.3, 0.3]
        prob = ODEProblem(model!, u0, tspan, p)
        sol = OrdinaryDiffEq.solve(prob)
        grid = sol(ts)
        equ = nlsolve((du, u) -> model!(du, u, p, 0.0), grid.u[end]).zero 
      
        PRlitt[Ni] = equ[5]/ equ[1]
        PRpel[Ni] = equ[5]/ equ[2]
        PClitt[Ni] = equ[5]/ equ[3]
        PCpel[Ni] = equ[5]/ equ[4]
         
    end
 return hcat(collect(Nvals), PRlitt, PRpel, PClitt, PCpel)
end

data= n_ratio_data()

println(n_ratio_data())
let
    data = n_ratio_data()
    ratios_plot = figure()
    plot(data[:,1], data[:,2], color = "blue")
    plot(data[:,1], data[:,3], color = "red")
    plot(data[:,1], data[:,4], color = "green")
    plot(data[:,1], data[:,5], color = "orange")
    ylabel("Biomass Ratios", fontsize = 14, fontweight=:bold)
    xlabel("Nutrient Concentration", fontsize = 14, fontweight=:bold)
    ylim(0.0,2.0)
    xlim(0.8, 1.66)
    return ratios_plot
end

## omnivory & coupling data (Structural EWS)
function structural_data(p)
    @unpack h_PC_litt, h_PC_pel, h_PR_litt, h_PR_pel, e_PC_litt, e_PC_pel, e_PR_litt, e_PR_pel, a_PR_litt, a_PR_pel, a_PC_litt,  a_PC_pel = p 
    Nvals = 0.185:0.005:1.66
    u0 = [ 0.5, 0.5, 0.3, 0.3, 0.3]
    tspan = (0.0, 10000.0)
    ts = range(9000, 10000, length = 1000)
    omnivory = zeros(length(Nvals))
    coupling = zeros(length(Nvals))


    for (Ni, Nval) in enumerate(Nvals)
        p = Par(n = Nval)
        prob = ODEProblem(model!, u0, tspan, p)
        sol = OrdinaryDiffEq.solve(prob)
        grid = sol(ts)
        eq = nlsolve((du, u) -> model!(du, u, p, 0.0), grid.u[end]).zero  
        RlittFR = ((e_PR_litt * a_PR_litt * eq[1] * eq[5])/(1 + a_PR_litt * h_PR_litt * eq[1] + a_PR_pel * h_PR_pel * eq[2] + a_PC_litt * h_PC_litt * eq[3] + a_PC_pel * h_PC_pel * eq[4])) 
        RpelFR = ((e_PR_pel * a_PR_pel * eq[2] * eq[5])/(1 + a_PR_litt * h_PR_litt * eq[1] + a_PR_pel * h_PR_pel * eq[2] + a_PC_litt * h_PC_litt * eq[3] + a_PC_pel * h_PC_pel * eq[4])) 
        ClittFR = ((e_PC_litt * a_PC_litt * eq[3] * eq[5])/(1 + a_PR_litt * h_PR_litt * eq[1] + a_PR_pel * h_PR_pel * eq[2] + a_PC_litt * h_PC_litt * eq[3] + a_PC_pel * h_PC_pel * eq[4])) 
        CpelFR = ((e_PC_pel * a_PC_pel * eq[4] * eq[5])/(1 + a_PR_litt * h_PR_litt * eq[1] + a_PR_pel * h_PR_pel * eq[2] + a_PC_litt * h_PC_litt * eq[3] + a_PC_pel * h_PC_pel * eq[4])) 
        omnivory[Ni] = (RlittFR + RpelFR)/ (RlittFR + RpelFR + ClittFR + CpelFR)
        coupling[Ni] = (RlittFR + ClittFR)/ (RlittFR + RpelFR + ClittFR + CpelFR)
    end
    return hcat(collect(Nvals), omnivory, coupling)
end

let
    data = structural_data(Par())
    structural_plot = figure()
    plot(data[:,1], data[:,2], color = "blue")
    plot(data[:,1], data[:,3], color = "red")
    ylabel("Structural Change", fontsize = 15)
    xlim(0.185, 1.66)
    ylim(0.0, 1.0)
    xlabel("Global Change", fontsize = 15)
    return structural_plot
end

function dynamical_data()
    Nvals = 0.185:0.005:1.66
    u0 = [0.5, 0.5, 0.3, 0.3, 0.3]
    tspan = (0.0, 10000.0)
    ts = range(0.0, 10000.0, length = 10000)
    cv_change = zeros(length(Nvals))

    for (Ni, Nval) in enumerate(Nvals)
        p = Par(n = Nval)
        prob = ODEProblem(model!, u0, tspan, p)
        sol = OrdinaryDiffEq.solve(prob)
        grid = sol(ts)
        std_change=std(grid[5,5000:10000])
        mean_change=mean(grid[5,5000:10000])
        cv_change[Ni] = std_change/mean_change

    end
    return hcat(collect(Nvals), cv_change)
end

let
    data = dynamical_data()
    dynamical_plot = figure()
    plot(data[:,1], data[:,2], color = "black")
    ylabel("CV", fontsize = 15)
    xlim(0.185, 1.66)
    ylim(0.0, 0.3)
    xlabel("Global Change", fontsize = 15)
    return dynamical_plot
end


## acf 
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
    Nvals = 0.185:0.05:1.66
    u0 = [0.5, 0.5, 0.3, 0.3, 0.3]
    tspan = (0.0, 5000.0)
    ts = range(0.0, 5000.0, length = 5000)
    acf = zeros(length(Nvals))

    for (Ni, Nval) in enumerate(Nvals)
        p = Par(n = Nval)
        prob = ODEProblem(model!, u0, tspan, p)
        sol = OrdinaryDiffEq.solve(prob)
        grid = sol(ts)
     
        acf[Ni] = autocor(grid[5,1:end], [1])[1]

    end
 return hcat(collect(Nvals), acf)
end


u0 = [0.5, 0.5, 0.3, 0.3, 0.3]
tspan = (0.0, 5000.0)
ts = range(0.0, 5000.0, length = 5000)
p = Par()

function max_acf_data()
    Nvals = 0.185:0.05:1.66
    u0 = [0.5, 0.5, 0.3, 0.3, 0.3]
    tspan = (0.0, 5000.0)
    ts = range(0.0, 5000.0, length = 5000)
    max_acf = zeros(length(Nvals))

   for (Ni, Nval) in enumerate(Nvals)
        p = Par(n = Nval)
        acf = autocor(grid[5,2000:5000], [1])
        max_acf[Ni] = maximum(acf)
  
    end
 return hcat(collect(Nvals), max_acf)
end

println(max_acf_data())
let
    data = acf_data()
    acf_plot = figure()
    plot(data[:,1], data[:,2], color = "black")
    ylabel("ACF Lag 1", fontsize = 15)
    xlabel("Global Change", fontsize = 15)
    ylim(0.75, 2.0)
    xlim(0.185, 1.66)
    return acf_plot
end

