using StartUpDG
using OrdinaryDiffEq
using Plots
using LinearAlgebra

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

using Trixi: CompressibleEulerEquations1D, flux, flux_ranocha, prim2cons, max_abs_speed_naive
using StaticArrays: SVector

equations = CompressibleEulerEquations1D(1.4)

function initial_condition(x, equations::CompressibleEulerEquations1D)
    (; gamma) = equations
    rho = 1 + (abs(x) < .4) 
    v1 = 0.0
    p = rho^gamma
    return SVector(rho, v1, p)
end

u0 = prim2cons.(initial_condition.(x, equations), equations)
f(u) = flux(u, 1, CompressibleEulerEquations1D(1.4))
fEC(u_i, u_j) = flux_ranocha(u_i, u_j, 1, CompressibleEulerEquations1D(1.4))
fEC(u_i, u_j) = flux_central(u_i, u_j, 1, CompressibleEulerEquations1D(1.4))
lambda(uM, uP) = max_abs_speed_naive(uM, uP, 1, CompressibleEulerEquations1D(1.4))

function rhs!(du, u, params, t)
    (; D, L, w, h, mapP) = params
    
    uM = u[[1, size(u, 1)], :]
    uP = uM[mapP]
    f_avg = @. 0.5 * (f(uM) + f(uP))
    fill!(du, zero(eltype(du)))
    for e in axes(u, 2)
        for i in axes(u, 1)
            u_i = u[i,e]
            for j in axes(u, 1)
                u_j = u[j,e]
                du[i, e] += 2 * D[i,j] * fEC(u_i, u_j)
            end
        end
    end    
    du .+= Diagonal(1 ./ w) * L * (@. (f_avg - f.(uM)) .* nx - 0.5 * lambda(uM, uP) .* (uP - uM))
    du .= -(2 / h) * du
end
tspan = (0.0, 4.0)
ode = ODEProblem(rhs!, u0, tspan, (; D, L, w, h, mapP, nx))
sol = solve(ode, Tsit5(), adaptive=false, dt = .1 / md.num_elements, 
            saveat = LinRange(tspan..., 100))

@gif for u in sol.u
    scatter(x, getindex.(u, 1), leg=false, ylims=extrema(getindex.(u0, 1)) .+ (-.1, .1))
end
