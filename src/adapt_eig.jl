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
    a_PR_litt = 0.3 
    a_PR_pel = 0.3
    aT_litt = 1.0
    aT_pel = 1.0
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

## using ForwardDiff for eigenvalue analysis, need to reassign model for just u
function adapt_model(u, AdaptPar, t)
    du = similar(u)
    adapt_model!(du, u, AdaptPar, t)
    return du
end

u0 = [ 0.5, 0.5, 0.3, 0.3, 0.3]
par = AdaptPar(T=30)
eq= nlsolve((du, u) -> adapt_model!(du, u, par, 0.0), u0).zero
Tis = 20:0.01:35
eqhold = fill(0.0,length(Tis),6)

## solving for equilibrium 
for i=1:length(Tis)
    par = AdaptPar(T=Tis[i])
   if i==1
     u0 = [0.5,0.5,0.5,0.5, 0.3]
   else 
     u0 = [eq[1], eq[2], eq[3], eq[4], eq[5]]
   end 
    eq = nlsolve((du, u) -> adapt_model!(du, u, par, 0.0), u0).zero

    eqhold[i,1] = Tis[i]
    eqhold[i,2:end] = eq
    println(eqhold[i,:])
end


# Function to calculate the jacobian at any point (with any model)
function jac(u, model, p)
    ForwardDiff.jacobian(u -> model(u, p, NaN), u)
end

# Functions to calculate maximum eigenvalue (real part) -- here temp is 25 degrees 
λ_stability(M) = maximum(real.(eigvals(M)))

adapt_jac = jac(eq, adapt_model, AdaptPar(T=25))

eigvals(adapt_jac)

λ_stability(adapt_jac)


# Check mark diagram
function T_maxeig_data()
    Tvals = 10:0.01:35
    max_eig = zeros(length(Tvals))

    for (Ti, Tval) in enumerate(Tvals)
        p = AdaptPar(T = Tval)
        max_eig[Ti] = λ_stability(jac(eq, adapt_model, p))
    end
    return hcat(collect(Tvals), max_eig)
end

let
    data = T_maxeig_data()
    maxeigen_plot = figure()
    plot(data[:,1], data[:,2], color = "black")
    ylabel("Re(λₘₐₓ)", fontsize = 15)
    xlim(10, 35)
    ylim(-0.35, 1.0)
    xlabel("Temperature", fontsize = 15)
    return maxeigen_plot
end