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
	return result
end

function backwardeuleronestep!(rhs, A, getb::Function, u_k, t, dt, linearsolver, atol)
	return backwardeuleronestep!(rhs, A, getb(t), u_k, dt, linearsolver, atol)
end

#this modifies rhs
function backwardeuleronestep!(rhs, A, b::Vector, u_k, dt, linearsolver, atol)
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

function backwardeulertwostep!(rhs, A, getb::Function, u_k, t, dt, linearsolver, atol, onestep=backwardeuleronestep!(rhs, A, getb, u_k, t, dt, linearsolver, atol))
	twostep1 = backwardeuleronestep!(rhs, A, getb, u_k, t, 0.5 * dt, linearsolver, atol)
	twostep = backwardeuleronestep!(rhs, A, getb, twostep1, t + 0.5 * dt, 0.5 * dt, linearsolver, atol)
	err = norm(onestep - twostep)
	if err < atol
		return twostep, dt, err < atol / 4
	else
		return twostep1, .5 * dt, false
	end
end

function adaptivebackwardeulerstep!(rhs, A, getb::Function, u_k, t, dt, linearsolver, atol, callback)
	u_new, laststeptime, increasestepsize = backwardeulertwostep!(rhs, A, getb, u_k, t, dt, linearsolver, atol)
	if laststeptime < dt#it couldn't take the step we asked, so try taking smaller steps
		laststepfailed = true
		elapsedtime = 0.0
		u_elapsedtime = u_k
		targetdt = laststeptime
		while elapsedtime < dt
			callback()
			if laststepfailed#if the last step failed, reuse u_new as the onestep part
				u_new, laststeptime, increasestepsize = backwardeulertwostep!(rhs, A, getb, u_elapsedtime, t + elapsedtime, targetdt, linearsolver, atol, u_new)
			else
				u_new, laststeptime, increasestepsize = backwardeulertwostep!(rhs, A, getb, u_elapsedtime, t + elapsedtime, targetdt, linearsolver, atol)
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
			targetdt = min(targetdt, dt - elapsedtime)#don't overshoot
		end
	end
	return u_new, laststeptime, increasestepsize
end

function backwardeulerintegrate(u0::T, A, b::Vector, dt0, t0::R, tfinal; kwargs...) where {T,R}
	function getb(t)
		return b
	end
	return backwardeulerintegrate(u0, A, getb, dt0, t0, tfinal; kwargs...)
end

function backwardeulerintegrate(u0::T, A, getb::Function, dt0, t0::R, tfinal; linearsolver=defaultlinearsolver, atol=1e-4, callback=()->nothing) where {T,R}
	us = T[u0]
	ts = R[t0]
	rhs = similar(u0)
	A = copy(A)
	dt = min(dt0, tfinal - t0)
	while ts[end] < tfinal
		solution, laststeptime, increasestepsize = adaptivebackwardeulerstep!(rhs, A, getb, us[end], ts[end], dt, linearsolver, atol, callback)
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

function backwardeulerintegrate(u0, tspan, Ss::Number, volumes::Vector, neighbors::Array{Pair{Int, Int}, 1}, areasoverlengths::Vector, conductivities::Vector, sources::Vector, dirichletnodes::Array{Int, 1}, dirichletheads::Vector, metaindex=i->i, logtransformconductivity=false; kwargs...)
	freenodes, nodei2freenodei = getfreenodes(length(u0), dirichletnodes)
	freenodei2nodei = Dict(zip(values(nodei2freenodei), keys(nodei2freenodei)))
	b = assembleb(neighbors, areasoverlengths, conductivities, sources, dirichletnodes, dirichletheads, metaindex, logtransformconductivity)
	scalebyvolume!(b, Ss * volumes, freenodei2nodei)
	getb(t) = b
	return backwardeulerintegrate(u0, tspan, getb, Ss, volumes, neighbors, areasoverlengths, conductivities, sources, dirichletnodes, dirichletheads, metaindex, logtransformconductivity; kwargs...)
end

function backwardeulerintegrate(u0, tspan, getb::Function, Ss::Number, volumes::Vector, neighbors::Array{Pair{Int, Int}, 1}, areasoverlengths::Vector, conductivities::Vector, sources::Vector, dirichletnodes::Array{Int, 1}, dirichletheads::Vector, metaindex=i->i, logtransformconductivity=false; dt0=1.0, kwargs...)
	A = assembleA(neighbors, areasoverlengths, conductivities, sources, dirichletnodes, dirichletheads, metaindex, logtransformconductivity)
	freenodes, nodei2freenodei = getfreenodes(length(u0), dirichletnodes)
	freenodei2nodei = Dict(zip(values(nodei2freenodei), keys(nodei2freenodei)))
	scalebyvolume!(A, Ss * volumes, freenodei2nodei)
	u0 = u0[freenodes]
	us, ts = backwardeulerintegrate(u0, A, getb, dt0, tspan[1], tspan[2]; kwargs...)
	us = map(x->freenodes2nodes(x, sources, dirichletnodes, dirichletheads)[1], us)
	return us, ts
end

function getcontinuoussolution(us::Vector{T}, ts::Vector) where {T <: AbstractArray}
	itp = Interpolations.interpolate((ts,), us, Interpolations.Gridded(Interpolations.Linear()))
	uc(t) = itp[t]
	return uc
end

function getcontinuoussolution(us::Vector{T}, ts::Vector, ::Type{Val{2}}) where {T <: AbstractArray}
	u = hcat(us...)
	itp = Interpolations.interpolate(([i for i = 1:size(u, 1)], ts), u, (Interpolations.NoInterp(), Interpolations.Gridded(Interpolations.Linear())))
	return itp
end

function adjointintegrate(getdgdu::Function, tspan, Ss::Number, volumes::Vector, neighbors::Array{Pair{Int, Int}, 1}, areasoverlengths::Vector, conductivities::Vector, sources::Vector, dirichletnodes::Array{Int, 1}, dirichletheads::Vector, metaindex=i->i, logtransformconductivity=false; kwargs...)
	A = assembleA(neighbors, areasoverlengths, conductivities, sources, dirichletnodes, dirichletheads, metaindex, logtransformconductivity)
	freenodes, nodei2freenodei = getfreenodes(length(volumes), dirichletnodes)
	freenodei2nodei = Dict(zip(values(nodei2freenodei), keys(nodei2freenodei)))
	scalebyvolume!(A, Ss * volumes, freenodei2nodei)
	return adjointintegrate(transpose(A), getdgdu, tspan; kwargs...)
end

#we use the notation from Strang's "Computational Science and Engineering"
#the adjoint equation is dλ/dt=-[df/du]ᵀ*λ-[dg/du]ᵀ with λ(T)=0, where f=b-Au
#we reformulate in terms of γ(t) = λ(T-t)
#dγ/dt(t)=[df/du(T-t)]ᵀ*γ+[dg/du(T-t)]ᵀ
#the goal is to facilitate the computation of the gradient of ∫g(u,t,p)dt by solving the adjoint equation
function adjointintegrate(A, getdgdu, tspan; dt0=1.0, kwargs...)
	gamma0 = zeros(size(A, 2))
	gammas, tsgamma = backwardeulerintegrate(gamma0, A, t->getdgdu(tspan[2] - t), dt0, tspan[1], tspan[2]; kwargs...)
	return reverse(gammas), reverse(tspan[2] - tsgamma)#return in terms of λ instead of γ
end

function gradientintegrate(lambdac::Function, du0dp, dgdp, dfdp::Function, tspan; kwargs...)
	I2, E2 = QuadGK.quadgk(t->dfdp(t) * lambdac(t), tspan...; kwargs...)
	return gradientintegrate(lambdac(0), du0dp, dgdp, I2, tspan; kwargs...)
end

function gradientintegrate(lambda0::Vector, du0dp, dgdp, integrateddfdplambda::Vector, tspan; kwargs...)
	I1, E1 = QuadGK.quadgk(t->dgdp(t), tspan...; kwargs...)
	dGdp = du0dp * lambda0 + I1 + integrateddfdplambda
	return dGdp
end
