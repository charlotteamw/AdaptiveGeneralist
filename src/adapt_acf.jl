using Distributed
using PyPlot


@everywhere using Parameters
@everywhere using DifferentialEquations
@everywhere using ForwardDiff
@everywhere using LinearAlgebra

@everywhere include("/Documents/AdaptiveGeneralist/src/adapt_stoch.jl")



## Calculating autocorrelations for one temperature -- no loop yet 
let
    u0 = [0.5, 0.5, 0.3, 0.3, 0.3]
    tspan = (0.0, 10000.0)
    p = AdaptPar(T=30.94, noise=0.05)
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


function acf_data()
    Tvals = 27.45:0.01:31.3
    acf_temp = zeros(length(Tvals))
    u0 = [ 0.5, 0.5, 0.3, 0.3, 0.3]
    tspan = (0.0, 10000.0)
    ts = range(5000, 6000, length = 1000)

   for (Ti, Tval) in enumerate(Tvals)
        p = AdaptPar(T = Tval)
        prob_stoch = SDEProblem(adapt_model!, stoch_adapt!, u0, tspan, p)
        sol_stoch = solve(prob_stoch, reltol = 1e-15)
        grid_sol= sol_stoch(ts)
        grid_sol.u
        grid_sol.t
        acf_temp[Ti] = autocor(grid_sol[5,1:end], [1])[1]
    end
 return hcat(collect(Tvals), acf_temp)
end

data = acf_data()
println(acf_data())


let
    data = acf_data()
    acf_plot = figure()
    plot(data[:,1], data[:,2], color = "black")
    ylabel("ACF at Lag = 1", fontsize = 14, fontweight=:bold)
    xlabel("Temperature", fontsize = 14, fontweight=:bold)
    ylim(-1.0,1.0)
    xlim(27.5, 31.1)
    return acf_plot
end


