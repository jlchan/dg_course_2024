using StartUpDG
using OrdinaryDiffEq
using Plots

N = 3
rd = RefElemData(Line(), SBP(), N)
md = MeshData(uniform_mesh(Line(), 64), rd; is_periodic = true)

x = md.x
h = x[end,1] - x[1,1]
nx = md.nx
w = rd.wq
D = rd.Dr
L = zeros(N+1, 2); L[1,1] = 1; L[end, end] = 1
mapP = md.mapP

u0(x) = exp(-10 * x^2)
# u0(x) = abs(x) < 0.5
u = u0.(x)

f(u) = u^2 / 2
fEC(u_i, u_j) = 1/6 * (u_i^2 + u_i * u_j + u_j^2)

function rhs!(du, u, params, t)
    (; D, L, w, h, mapP) = params
    
    uM = u[[1, size(u, 1)], :]
    uP = uM[mapP]
    f_avg = @. 0.5 * (f(uM) + f(uP))
    # f_avg = @. fEC(uM, uP) # need this if lambda = 0
    du .= zero(eltype(du))
    for e in axes(u, 2)
        for i in axes(u, 1)
            u_i = u[i,e]
            for j in axes(u, 1)
                u_j = u[j,e]
                du[i, e] += 2 * D[i,j] * fEC(u_i, u_j)
            end
        end
    end
    lambda = @. max(abs(uM), abs(uP))
    du .+= Diagonal(1 ./ w) * L * ((f_avg - f.(uM)) .* nx - 0.5 * lambda .* (uP - uM))
    du .= -(2 / h) * du
end
tspan = (0.0, 5.0)
ode = ODEProblem(rhs!, u, tspan, (; D, L, w, h, mapP, nx))
sol = solve(ode, Tsit5(), adaptive=false, dt = .05 / md.num_elements, #abstol=1e-12, reltol=1e-12,
            saveat = LinRange(tspan..., 50))

@gif for u in sol.u
    scatter(x, u, leg=false, ylims=(-.5, 1.5))
end
# scatter(x, sol.u[end], leg=false)
