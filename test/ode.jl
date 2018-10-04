using Test
import FiniteVolume
import Interpolations
import LinearAlgebra
import QuadGK
import SparseArrays

#solve dy/dt=-A*y with y(0)=[1, 1, 1], A=diagm([1, 2, 3])
#the analytical solution is y=[e^-t, e^(-2*t), e^(-3*t)]
v = [1.0, 2.0, 3.0]
A = SparseArrays.sparse(LinearAlgebra.Diagonal(v))
b = zeros(3)
y0 = [1.0, 1.0, 1.0]
ys, ts = FiniteVolume.backwardeulerintegrate(y0, A, b, 0.0001, 0.0, 2.0; atol=1e-8)
for i = 1:length(ys)
	for j = 1:length(ys[i])
		@test isapprox(ys[i][j], exp(-v[j] * ts[i]); atol=1e-4)
	end
end

#solve dy/dt=y+1, y(0)=0, which has solution y(t)=e^x-1
v = [-1.0]
A = SparseArrays.sparse(LinearAlgebra.Diagonal(v))
b = [1.0]
y0 = [0.0]
ys, ts = FiniteVolume.backwardeulerintegrate(y0, A, b, 0.0001, 0.0, 1.0; atol=1e-8)
for i = 1:length(ys)
	@test isapprox(ys[i][1], exp(ts[i]) - 1; atol=1e-4)
end

#solve dy/dt=Ay where A=[.5 -1; 1 -1], which has solution y(t) given below
y(t; c1=1, c2=2) = c1 * exp(-t / 4) * ([1, 0.75] * cos(sqrt(7) * t / 4) - [0, -sqrt(7) / 4] * sin(sqrt(7) * t / 4)) + c2 * exp(-t / 4) * ([1, 0.75] * sin(sqrt(7) * t / 4) + [0, -sqrt(7) / 4] * cos(sqrt(7) * t / 4))
A = -[.5 -1; 1 -1]
b = zeros(2)
y0 = y(0)
linearsolver(A, b, x0) = A \ b
ys, ts = FiniteVolume.backwardeulerintegrate(y0, A, b, 1e-4, 0.0, 1e2; atol=1e-8, linearsolver=linearsolver)
for i = 1:length(ys)
	@test isapprox(ys[i], y(ts[i]); atol=1e-4)
end
