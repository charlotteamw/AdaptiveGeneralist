include("generalist_module.jl")
using Parameters
using LinearAlgebra
using ForwardDiff
using PyPlot
using DifferentialEquations
using NLsolve


## plotting CV 
function cv_data()
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
    data = cv_data()
    cv_plot = figure()
    plot(data[:,1], data[:,2], color = "black")
    ylabel("CV", fontsize = 15)
    xlim(0.185, 1.66)
    ylim(0.0, 0.3)
    xlabel("Global Change", fontsize = 15)
    return cv_plot
end


## acf calculation
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
