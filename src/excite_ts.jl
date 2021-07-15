include("generalist_module.jl")



## plot time series 
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