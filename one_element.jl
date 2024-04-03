using StartUpDG
using OrdinaryDiffEq
using Plots

N = 25
rd = RefElemData(Line(), SBP(), N)

x = rd.r
D = rd.Dr
w = rd.wq

u0(x) = sin(pi * x)
u = u0.(x)
function rhs!(du, u, params, t)
    (; D, w) = params
    u_avg = 0.5 * (u[1] + u[end])
    du .= D * u
    du[1] += -(u_avg - u[1]) / w[1]
    du[end] += (u_avg - u[end]) / w[end]
    @. du = -du
end
ode = ODEProblem(rhs!, u, (0.0, 2.0), (; D, w))
sol = solve(ode, Tsit5(), abstol=1e-12, reltol=1e-12)

@gif for u in sol.u
    scatter(x, u)
end