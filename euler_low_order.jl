using StartUpDG
using OrdinaryDiffEq
using Plots
using LinearAlgebra, SparseArrays

N = 3
rd = RefElemData(Line(), SBP(), N)
md = MeshData(uniform_mesh(Line(), 32), rd; is_periodic = true)

x = md.x
Q = rd.M * rd.Dr
Qskew = 0.5 * (Q - Q')
Qskew = 0.5 * spdiagm(-1 => -ones(rd.N), 1 => ones(rd.N))
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
                Q_ij = Qskew[i,j]
                if abs(Q_ij) > 100 * eps()
                    n_ij = Q_ij / abs(Q_ij)
                    #du[i, e] += 2 * Qskew[i,j] * fEC(u_i, u_j)
                    # du[i, e] += 2 * Q_ij * 0.5 * (f(u_i) + f(u_j)) - 
                    #     abs(Q_ij) * lambda(u_i, u_j) * (u_j - u_i)
                    du[i, e] += 2 * abs(Q_ij) * flux_lax_friedrichs(u_i, u_j, SVector(n_ij), CompressibleEulerEquations1D(1.4))
                end
            end
        end
    end    
    du .= -rd.M \ (du ./ md.J)
end
tspan = (0.0, .20)
ode = ODEProblem(rhs!, u0, tspan, (; Qskew, ETr, rd, md))
sol = solve(ode, SSPRK43(), abstol=1e-8, reltol=1e-5, 
            callback=AliveCallback(alive_interval=100),
            saveat = LinRange(tspan..., 50))

@gif for u in sol.u
    scatter(x, getindex.(u, 1), leg=false, ylims=extrema(getindex.(u0, 1)) .+ (-.1, .1))
end
