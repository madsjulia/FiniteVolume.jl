import DifferentialEquations

function diagonalupdate!(A, increment)
	rows = rowvals(A)
	vals = nonzeros(A)
	m, n = size(A)
	for i = 1:n
		for j in nzrange(A, i)
			if rows[j] == i#update the diagonal
				vals[j] += increment
			end
		end
	end
end

function defaultlinearsolver(A, b)
	M = PyAMG.aspreconditioner(PyAMG.RugeStubenSolver(A))
	result, ch = IterativeSolvers.cg(A, b; Pl=M, log=true, maxiter=400)
	return result
end

#this modifies rhs and (maybe) A
function backwardeuleronestep!(rhs, A, b, u_k, dt, linearsolver, rtol, mindt)
	@. rhs = b + u_k / dt
	diagonalupdate!(A, 1 / dt)
	onestep = linearsolver(A, rhs)
	diagonalupdate!(A, -1 / dt)
	return onestep
end

function backwardeulertwostep!(rhs, A, b, u_k, dt, linearsolver, rtol, mindt, onestep=Float64[])
	#u_{k+1} - u_k = dt * (b - A * u_{k+1})
	#(I / dt + A) u_{k+1} = u_k / dt + b
	if dt < mindt
		error("time step too small $dt < $mindt")
	end
	if length(onestep) == 0
		onestep = backwardeuleronestep!(rhs, A, b, u_k, dt, linearsolver, rtol, mindt)
	end
	twostep1 = backwardeuleronestep!(rhs, A, b, u_k, 0.5 * dt, linearsolver, rtol, mindt)
	twostep = backwardeuleronestep!(rhs, A, b, twostep1, 0.5 * dt, linearsolver, rtol, mindt)
	err = norm(onestep - twostep) / norm(onestep)
	@show dt, err
	if err < rtol
		return twostep, dt
	else
		return twostep1, .5 * dt
	end
end

function adaptivebackwardeulerstep!(rhs, A, b, u_k, dt, linearsolver, rtol, mindt)
	u_new, laststeptime = backwardeulertwostep!(rhs, A, b, u_k, dt, linearsolver, rtol, mindt)
	if laststeptime < dt#it couldn't take the step we asked, so try taking smaller steps
		elapsedtime = 0.0
		u_elapsedtime = u_k
		targetdt = laststeptime
		while elapsedtime < dt
			u_new, laststeptime = backwardeulertwostep!(rhs, A, b, u_elapsedtime, targetdt, linearsolver, rtol, mindt, u_new)
			if laststeptime == targetdt
				elapsedtime += laststeptime
				u_elapsedtime = u_new
				targetdt = min(dt - elapsedtime, 2 * laststeptime)
			elseif laststeptime < targetdt
				targetdt = min(dt - elapsedtime, laststeptime)
			else
				error("Code is broken -- laststeptime should never be less than targetdt")
			end
		end
	end
	return u_new, laststeptime
end

function backwardeulerintegrate(u0::T, A, b, dt0, t0::R, tfinal; linearsolver=defaultlinearsolver, rtol=1e-4, mindt=sqrt(eps(Float64))) where {T,R}
	us = T[u0]
	ts = R[t0]
	rhs = similar(u0)
	A = copy(A)
	dt = min(dt0, tfinal - t0)
	while ts[end] < tfinal
		#@time solution = backwardeulerstep!(rhs, A, b, us[end], dt0, linearsolver, rtol)
		#@time solution, dt = backwardeulertwostep!(rhs, A, b, us[end], dt0, linearsolver, rtol, mindt)
		@time solution, laststeptime = adaptivebackwardeulerstep!(rhs, A, b, us[end], dt, linearsolver, rtol, mindt)
		push!(us, solution)
		push!(ts, ts[end] + dt)
		@show ts[end]
		newdt = min(tfinal - ts[end], 2 * laststeptime)
		dt = newdt
	end
	return us, ts
end

function backwardeulerintegrate(u0, tspan, neighbors::Array{Pair{Int, Int}, 1}, areasoverlengths::Vector, conductivities::Vector, sources::Vector, dirichletnodes::Array{Int, 1}, dirichletheads::Vector, metaindex=i->i; dt0=1.0, kwargs...)
	A = assembleA(neighbors, areasoverlengths, conductivities, sources, dirichletnodes, dirichletheads, metaindex)
	b = assembleb(neighbors, areasoverlengths, conductivities, sources, dirichletnodes, dirichletheads, metaindex)
	freenodes, _ = getfreenodes(length(u0), dirichletnodes)
	u0 = u0[freenodes]
	us, ts = backwardeulerintegrate(u0, A, b, dt0, tspan[1], tspan[2]; kwargs...)
	us = map(x->freenodes2nodes(x, sources, dirichletnodes, dirichletheads)[1], us)
	return us, ts
end

function integrate(u0, tspan, neighbors::Array{Pair{Int, Int}, 1}, areasoverlengths::Vector, conductivities::Vector, sources::Vector, dirichletnodes::Array{Int, 1}, dirichletheads::Vector, metaindex=i->i; solver=DifferentialEquations.Tsit5())
	A = assembleA(neighbors, areasoverlengths, conductivities, sources, dirichletnodes, dirichletheads, metaindex)
	b = assembleb(neighbors, areasoverlengths, conductivities, sources, dirichletnodes, dirichletheads, metaindex)
	freenodes, _ = getfreenodes(length(u0), dirichletnodes)
	u0 = u0[freenodes]
	storage = similar(u0)
	setupdu = (t, u, du)->begin
		A_mul_B!(storage, A, u)
		@. du = b - storage
	end
	prob = DifferentialEquations.ODEProblem(setupdu, u0, tspan)
	sol = DifferentialEquations.solve(prob, solver)
	return sol
end
