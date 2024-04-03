using StartUpDG
using OrdinaryDiffEq
using Plots

N = 3
rd = RefElemData(Line(), SBP(), N)
md = MeshData(uniform_mesh(Line(), 128), rd; is_periodic = true)

x = md.x
h = x[end,1] - x[1,1]
nx = md.nx
w = rd.wq
D = rd.Dr
L = zeros(N+1, 2); L[1,1] = 1; L[end, end] = 1
mapP = md.mapP

# u0(x) = exp(-25 * x^2)
u0(x) = abs(x) < 0.5
u = u0.(x)
function rhs!(du, u, params, t)
    (; D, L, w, h, mapP) = params
    
    uM = u[[1, size(u, 1)], :]
    uP = uM[mapP]
    u_avg = 0.5 * (uM + uP) 
    du .= D * u + Diagonal(1 ./ w) * L * ((u_avg - uM) .* nx - 0.5 * (uP - uM))
    du .= -(2 / h) * du
end
tspan = (0.0, 2.0)
ode = ODEProblem(rhs!, u, tspan, (; D, L, w, h, mapP, nx))
sol = solve(ode, Tsit5(), adaptive=false, dt = .1 / md.num_elements, #abstol=1e-12, reltol=1e-12,
            saveat = LinRange(tspan..., 50))

@gif for u in sol.u
    scatter(x, u, leg=false, ylims=(-.5, 1.5))
end
