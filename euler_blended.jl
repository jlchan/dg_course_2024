using StartUpDG
using OrdinaryDiffEq
using Plots
using LinearAlgebra, SparseArrays

N = 3
rd = RefElemData(Line(), SBP(), N)
md = MeshData(uniform_mesh(Line(), 128), rd; is_periodic = true)

x = md.x
Q = rd.M * rd.Dr
Qskew = (Q - Q')
Qskew = diagm(-1 => -ones(N), 1 => ones(N))
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

    du_high = similar(du[:,1])
    du_low = similar(du[:,1])
    for e in axes(u, 2)

        fill!(du_high, zero(eltype(du)))
        fill!(du_low, zero(eltype(du)))

        for i in axes(u, 1)
            u_i = u[i,e]
            for j in axes(u, 1)
                u_j = u[j,e]
                du_high[i] += Qskew[i,j] * fEC(u_i, u_j)

                S_ij = Qskew[i,j]
                if abs(S_ij) > 100 * eps()
                    n_ij = S_ij / abs(S_ij)
                    du_low[i] += abs(S_ij) * flux_lax_friedrichs(u_i, u_j, SVector(n_ij), 
                                                                 CompressibleEulerEquations1D(1.4))
                end
            end
        end

        θ = .99
        # if minimum(getindex.(u[:, e], 1)) < .001
        #     θ = 0
        # end
        @. du[:, e] .+= θ * du_high + (1 - θ) * du_low
    end    
    du .= -rd.M \ (du ./ md.J)
end
tspan = (0.0, .20)
ode = ODEProblem(rhs!, u0, tspan, (; Qskew, ETr, rd, md))
sol = solve(ode, SSPRK43(), abstol=1e-9, reltol=1e-6, 
            callback=AliveCallback(alive_interval=100),
            saveat = LinRange(tspan..., 50))

u = sol.u[end]            
scatter(x, getindex.(u, 1), leg=false)
# @gif for u in sol.u
#     scatter(x, getindex.(u, 1), leg=false, ylims=extrema(getindex.(u0, 1)) .+ (-.1, .1))
# end
