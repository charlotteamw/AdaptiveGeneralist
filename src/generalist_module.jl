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
