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


## Generalist Omnivory Module with Temp Dependent Attack Rates (a_PC_litt => aPC in littoral zone; a_PC_pel => aPC in pelagic zone)
# unpack is the passing of the mutable parameters

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

## Calculate Equilibrium
u0 = [ 0.5, 0.5, 0.3, 0.3, 0.3]
p = AdaptPar(T=30)
eq= nlsolve((du, u) -> adapt_model!(du, u, p, 0.0), u0).zero
Tis = 29:0.01:32.0
eqhold = fill(0.0,length(Tis),6)
for i=1:length(Tis)
    par = AdaptPar(T=Tis[i])
   if i==1
     u0 = [ 0.5928342081370711,
     0.6974050738463745,
     0.10605886832238756,
     0.003509224701950255,
     0.027302094900570676
    ]
   else 
     u0 = [eq[1], eq[2], eq[3], eq[4], eq[5]]
   end 
    eq = nlsolve((du, u) -> adapt_model!(du, u, par, 0.0), u0).zero
    eqhold[i,1] = Tis[i]
    eqhold[i,2:end] = eq
    println(eqhold[i,:])
end


function T_omniv_data()
        Tvals = 27.5:0.01:31.5
        omnivory = zeros(length(Tvals))

        a_PC_litt = ifelse(T < Topt_litt,  
        aT_litt * exp(-((T - Topt_litt)/(2*σ))^2), 
        aT_litt * (1 - ((T - (Topt_litt))/((Topt_litt) - Tmax_litt))^2)
        )
        
        a_PC_pel = ifelse(T < Topt_pel, 
        aT_pel * exp(-((T - Topt_pel)/(2*σ))^2),
        aT_pel * (1 - ((T - (Topt_pel))/((Topt_pel) - Tmax_pel))^2) 
        )
        
        for (Ti, Tval) in enumerate(Tvals)
            p = AdaptPar(T = Tval)
            omnivory[Ti] = ((a_PR_litt * e_PR * eqhold[1,6] * eqhold[1,2])/(1 + h_PR * a_PR_litt * eqhold[1,2]) + (a_PR_pel * e_PR * eqhold[1,6] * eqhold[1,3])/(1 + h_PR * a_PR_pel * eqhold[1,3]) )/ ((a_PR_litt * e_PR * eqhold[1,6] * eqhold[1,2])/(1 + h_PR * a_PR_litt * eqhold[1,2]) + (a_PR_pel * e_PR * eqhold[1,6] * eqhold[1,3])/(1 + h_PR * a_PR_pel * eqhold[1,3]) + (a_PC_litt * e_PC * eqhold[1,6] * eqhold[1,4])/(1 + h_PC * a_PC_litt * eqhold[1,4]) + (a_PC_pel * e_PC * eqhold[1,6] * eqhold[1,5])/(1 + h_PC * a_PC_pel * eqhold[1,5]))
        end
        return hcat(collect(Tvals), omnivory)
 end

    

 let
        data = T_omniv_data()
        omnivory_plot = figure()
        plot(data[:,1], data[:,2], color = "black")
        ylabel("Omnivory", fontsize = 15)
        xlim(27.5, 31.5)
        ylim(0.0, 1.0)
        xlabel("Temperature", fontsize = 15)
        return omnivory_plot
end
## time span for ode solver when we use it
# u0 is initial value of the initial value problem 



# this the ode version solving in time
tspan = (0.0, 1000.0)
u0 = [ 0.5, 0.5, 0.3, 0.3, 0.3]
par = AdaptPar(T=30)
prob = ODEProblem(adapt_model!, u0, tspan, par)
sol = OrdinaryDiffEq.solve(prob)
equil= nlsolve((du, u) -> adapt_model!(du, u, par, 0.0), sol.u[end]).zero


# this is just solving algebraically where the odes =0 --> looks just for equilibrium
# u0 is this initial valu you use to seed the nlsolve, if you use sol you use the last
# solution out of the ode solver, lets not do that now
eq= nlsolve((du, u) -> adapt_model!(du, u, par, 0.0), u0).zero