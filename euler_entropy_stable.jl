using StartUpDG
using OrdinaryDiffEq
using Plots
using LinearAlgebra, SparseArrays
using Trixi

N = 1
rd = RefElemData(Line(), SBP(), N)
md = MeshData(uniform_mesh(Line(), 512), rd; is_periodic = true)

x = md.x
Q = rd.M * rd.Dr
Qskew = 0.5 * (Q - Q')
ETr = spzeros(N+1, 2)
ETr[1,1] = 1 
ETr[end, end] = 1

using Trixi: CompressibleEulerEquations1D, flux, flux_ranocha, prim2cons, max_abs_speed_naive
using StaticArrays: SVector

equations = CompressibleEulerEquations1D(1.4)

function initial_condition(x, equations::CompressibleEulerEquations1D)
    (; gamma) = equations
    rho = .01 + (abs(x) < 0.4)
    v1 = 0.0
    p = rho^gamma
    return SVector(rho, v1, p)
end

u0 = prim2cons.(initial_condition.(x, equations), equations)
f(u) = flux(u, 1, CompressibleEulerEquations1D(1.4))
fEC(u_i, u_j) = flux_ranocha(u_i, u_j, 1, CompressibleEulerEquations1D(1.4))
lambda(uM, uP) = max_abs_speed_naive(uM, uP, 1, CompressibleEulerEquations1D(1.4))

function rhs!(du, u, params, t)
    (; Qskew, ETr, rd, md) = params    

    fill!(du, zero(eltype(du)))

    uM = u[rd.Fmask, :]
    uP = uM[md.mapP]
    du .= ETr * (@. 0.5 * (f(uM) + f(uP)) * md.nx - 0.5 * lambda(uM, uP) * (uP - uM))

    for e in axes(u, 2)
        for i in axes(u, 1)
            u_i = u[i,e]
            for j in axes(u, 1)
                u_j = u[j,e]
                du[i, e] += 2 * Qskew[i,j] * fEC(u_i, u_j)
            end
        end
    end    
    du .= -rd.M \ (du ./ md.J)
end
tspan = (0.0, .2)
ode = ODEProblem(rhs!, u0, tspan, (; Qskew, ETr, rd, md, equations))
sol = solve(ode, SSPRK43(), abstol=1e-9, reltol=1e-6,
            saveat = LinRange(tspan..., 100), 
            callback=AliveCallback(alive_interval=100))

# @gif for u in sol.u
#     scatter(x, getindex.(u, 1), leg=false, ylims=extrema(getindex.(u0, 1)) .+ (-.1, .1))
# end

@gif for u in [sol.u[end]]
    # scatter(x, getindex.(u, 1), leg=false, ylims=extrema(getindex.(u0, 1)) .+ (-.1, .1))
    q = cons2prim.(u, equations)
    u_plus_c = @. getindex(q, 2) + sqrt(equations.gamma * getindex(q,3) / getindex(q,1)) # u + sqrt(gamma * p / rho)
    u_minus_c = @. getindex(q, 2) - sqrt(equations.gamma * getindex(q,3) / getindex(q,1)) # u + sqrt(gamma * p / rho)
    scatter(x, getindex.(u_plus_c, 1), leg=false)
    scatter!(x, getindex.(u_minus_c, 1), leg=false)
end
