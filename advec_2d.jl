using StartUpDG
using OrdinaryDiffEq
using Plots

rd = RefElemData(Tri(), SBP(), N=3)
md = MeshData(uniform_mesh(Tri(), 32), rd)

(; x, y) = md
u = @. sin(pi * x) * sin(pi * y)

function rhs!(du, u, p, t)
    (; rd, md) = p

    uM = rd.Vf * u
    uP = uM[md.mapP]
    u_fluxJ = @. 0.5 * (uP + uM) * md.nxJ

    # dudx = dr/dx * du/dr + ds/dx * du/ds 
    dudr = rd.Dr * u
    duds = rd.Ds * u
    dudxJ = @. md.rxJ * dudr + md.sxJ * duds

    du .= dudxJ + rd.LIFT * (u_fluxJ - uM .* md.nxJ)
    du ./= -md.J
end

tspan = (0.0, 2.0)
ode = ODEProblem(rhs!, u, tspan, (; rd, md))
sol = solve(ode, Tsit5(), abstol=1e-12, reltol=1e-10,
            saveat = LinRange(tspan..., 50))

@gif for u in sol.u
    scatter(md.xyz..., zcolor=u, leg=false, ylims=(-.5, 1.5))
end


