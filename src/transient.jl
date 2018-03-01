import DifferentialEquations

function diagonalupdate!(A::Array{T, 2}, increment) where {T}
	for i = 1:size(A, 1)
		A[i, i] += increment
	end
end

function scalebyvolume!(b::Vector, volumes, freenodei2nodei)
	for i = 1:length(b)
		b[i] /= volumes[freenodei2nodei[i]]
	end
end

function scalebyvolume!(A::SparseMatrixCSC, volumes, freenodei2nodei)
	rows = rowvals(A)
	vals = nonzeros(A)
	m, n = size(A)
	for i = 1:n
		for j in nzrange(A, i)
			vals[j] /= volumes[freenodei2nodei[rows[j]]]
		end
	end
end

function diagonalupdate!(A::SparseMatrixCSC, increment)
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

function defaultlinearsolver(A, b, x0)
	result = copy(x0)
	result, ch = IterativeSolvers.cg!(result, A, b; log=true, maxiter=100)
	if !ch.isconverged#if it didn't converge without preconditioning, try it with preconditioning
		M = PyAMG.aspreconditioner(PyAMG.RugeStubenSolver(A))
		result, ch = IterativeSolvers.cg!(result, A, b; Pl=M, log=true, maxiter=100)
	end
	#result = A \ b
	return result
end

function backwardeuleronestep!(rhs, A, getb::Function, u_k, t, dt, linearsolver, atol, mindt)
	return backwardeuleronestep!(rhs, A, getb(t), u_k, dt, linearsolver, atol, mindt)
end

#this modifies rhs
function backwardeuleronestep!(rhs, A, b::Vector, u_k, dt, linearsolver, atol, mindt)
	#u_{k+1} - u_k = dt * (b - A * u_{k+1})
	#(I / dt + A) u_{k+1} = u_k / dt + b
	if dt <= 0
		error("time step must be positive")
	end
	@. rhs = b + u_k / dt
	diagonalupdate!(A, 1 / dt)
	onestep = linearsolver(A, rhs, u_k)
	diagonalupdate!(A, -1 / dt)
	return onestep
end

function backwardeulertwostep!(rhs, A, getb::Function, u_k, t, dt, linearsolver, atol, mindt, onestep=backwardeuleronestep!(rhs, A, getb, u_k, t, dt, linearsolver, atol, mindt))
	twostep1 = backwardeuleronestep!(rhs, A, getb, u_k, t, 0.5 * dt, linearsolver, atol, mindt)
	twostep = backwardeuleronestep!(rhs, A, getb, twostep1, t + 0.5 * dt, 0.5 * dt, linearsolver, atol, mindt)
	err = norm(onestep - twostep)
	if err < atol
		return twostep, dt, err < atol / 4
	else
		return twostep1, .5 * dt, false
	end
end

function adaptivebackwardeulerstep!(rhs, A, getb::Function, u_k, t, dt, linearsolver, atol, mindt)
	u_new, laststeptime, increasestepsize = backwardeulertwostep!(rhs, A, getb, u_k, t, dt, linearsolver, atol, mindt)
	if laststeptime < dt#it couldn't take the step we asked, so try taking smaller steps
		laststepfailed = true
		elapsedtime = 0.0
		u_elapsedtime = u_k
		targetdt = laststeptime
		while elapsedtime < dt
			if laststepfailed#if the last step failed, reuse u_new as the onestep part
				u_new, laststeptime, increasestepsize = backwardeulertwostep!(rhs, A, getb, u_elapsedtime, t + elapsedtime, targetdt, linearsolver, atol, mindt, u_new)
			else
				u_new, laststeptime, increasestepsize = backwardeulertwostep!(rhs, A, getb, u_elapsedtime, t + elapsedtime, targetdt, linearsolver, atol, mindt)
			end
			if laststeptime == targetdt
				elapsedtime += laststeptime
				u_elapsedtime = u_new
				if increasestepsize
					targetdt = 2 * laststeptime
				end
				laststepfailed = false
			elseif laststeptime < targetdt
				targetdt = laststeptime
				laststepfailed = true
			else
				error("Code is broken -- laststeptime should never be greater than targetdt")
			end
		end
	end
	return u_new, laststeptime, increasestepsize
end

function backwardeulerintegrate(u0::T, A, b::Vector, dt0, t0::R, tfinal; linearsolver=defaultlinearsolver, atol=1e-4, mindt=sqrt(eps(Float64))) where {T,R}
	function getb(t)
		return b
	end
	return backwardeulerintegrate(u0, A, getb, dt0, t0, tfinal; linearsolver=linearsolver, atol=atol, mindt=mindt)
end

function backwardeulerintegrate(u0::T, A, getb::Function, dt0, t0::R, tfinal; linearsolver=defaultlinearsolver, atol=1e-4, mindt=sqrt(eps(Float64))) where {T,R}
	us = T[u0]
	ts = R[t0]
	rhs = similar(u0)
	A = copy(A)
	dt = min(dt0, tfinal - t0)
	while ts[end] < tfinal
		solution, laststeptime, increasestepsize = adaptivebackwardeulerstep!(rhs, A, getb, us[end], ts[end], dt, linearsolver, atol, mindt)
		push!(us, solution)
		push!(ts, ts[end] + dt)
		if increasestepsize
			newdt = min(tfinal - ts[end], 2 * laststeptime)
		else
			newdt = min(tfinal - ts[end], laststeptime)
		end
		dt = newdt
	end
	return us, ts
end

function backwardeulerintegrate(u0, tspan, Ss::Number, volumes::Vector, neighbors::Array{Pair{Int, Int}, 1}, areasoverlengths::Vector, conductivities::Vector, sources::Vector, dirichletnodes::Array{Int, 1}, dirichletheads::Vector, metaindex=i->i; dt0=1.0, kwargs...)
	A = assembleA(neighbors, areasoverlengths, conductivities, sources, dirichletnodes, dirichletheads, metaindex)
	b = assembleb(neighbors, areasoverlengths, conductivities, sources, dirichletnodes, dirichletheads, metaindex)
	freenodes, nodei2freenodei = getfreenodes(length(u0), dirichletnodes)
	freenodei2nodei = Dict(zip(values(nodei2freenodei), keys(nodei2freenodei)))
	scalebyvolume!(A, Ss * volumes, freenodei2nodei)
	scalebyvolume!(b, Ss * volumes, freenodei2nodei)
	u0 = u0[freenodes]
	us, ts = backwardeulerintegrate(u0, A, b, dt0, tspan[1], tspan[2]; kwargs...)
	us = map(x->freenodes2nodes(x, sources, dirichletnodes, dirichletheads)[1], us)
	return us, ts
end
