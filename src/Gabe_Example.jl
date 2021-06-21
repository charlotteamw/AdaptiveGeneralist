using Parameters
using DifferentialEquations
using PyPlot
pygui(true)
@with_kw mutable struct Par
    r = 2.0
    K = 3.0
    a = 1.1
    h = 0.8
    e = 0.7
    m = 0.4
    σ = 0.1
end
# Standard inplace version
function roz_mac!(du, u, p, t,)
    @unpack r, K, a, h, e, m = p
    R, C = u
    du[1] = r * R * (1 - R / K) - a * R * C / (1 + a * h * R)
    du[2] = e * a * R * C / (1 + a * h * R) - m * C
    return
end
t_span = (0.0, 1000.0)
u0 = [2.0, 1.0]
par = Par()
prob = ODEProblem(roz_mac!, u0, t_span, par)
sol = solve(prob)
# The returned object in `sol` is actually really fancy (partially thanks to yours truly ;))
## These give the underlying time steps used by the solver ... so not regular for any solver worth its salt
sol.t
sol.u
plot(sol.t, sol.u, "o")
xlim(0, 10)
## Lets say we want the solutions at only certain time steps (just make sure it is inside of `t_span`! extrapolation is the road to sadness)
ts = range(0, 1000, length = 200)
## Then we just need to *call* the solution object
grid_sol = sol(ts)
grid_sol.t
grid_sol.u
plot(sol.t, sol.u, "o")
plot(grid_sol.t, grid_sol.u, "o")
xlim(0, 10)