#--------------------------------#
#         House-keeping          #
#--------------------------------#

using Distributed
using Distributions
using Compat.Dates

#--------------------------------#
#         Initialization         #
#--------------------------------#

# Number of cores/workers
addprocs(Sys.CPU_THREADS-1)
# Utility function
γ = 8; β = 0.9;
# Number of periods
T=20;
# Grid for x
nx    = 1001; xmin  = 0.0; xmax  = 1.0;
xgrid = zeros(nx);
size = nx;
xstep = (xmax - xmin) /(size - 1);
for i = 1:nx
  xgrid[i] = xmin + (i-1)*xstep;
end
# Value function
Vrm         = zeros(T, nx);
V_tomorrow  = zeros(nx)

#--------------------------------#
#     Structure and function     #
#--------------------------------#

# Data structure of state and exogenous variables
@everywhere struct modelState
  ix::Int64
  nx::Int64
  T::Int64
  age::Int64
  xgrid::Vector{Float64}
  β::Float64
  V::Array{Float64,1}
  γ::Float64
end

@everywhere function value(currentState::modelState)
  ix      = currentState.ix
  nx      = currentState.nx
  T       = currentState.T
  age     = currentState.age
  xgrid   = currentState.xgrid
  β    = currentState.β
  V     = currentState.V
  γ  = currentState.γ

  VV=0.0

  for ixp = 1:nx
    tmr = 0.0
      if (age<T)
        tmr = V[ixp]
      end

      cons = xgrid[ix]-xgrid[ixp]
      utility = ((xgrid[ix]-xgrid[ixp])^(1-γ)/(1-γ))+β*tmr

      if(cons <= 0)
        utility = -10.0^(5);
      end

      if(utility >= VV)
        VV = utility;
      end

    utility = 0.0;
  end

return(VV);

end

#--------------------------------#
#     Cake-eating computation     #
#--------------------------------#

print(" \n")
print("Cake eating computation with PMAP: \n")
print(" \n")

start = Dates.unix2datetime(time())

for age = T:-1:1
  pars = [modelState(ix,nx,T,age,xgrid,β,V_tomorrow,γ) for ix in 1:nx];
  s = pmap(value,pars)

  for ix = 1:nx
    Vrm[age, ix] = s[ix]
    V_tomorrow[ix] = s[ix]
  end

  finish = convert(Int, Dates.value(Dates.unix2datetime(time())- start))/1000;
  print("Age: ", age, ". Time: ", finish, " seconds. \n")

end

print("\n")
finish = convert(Int, Dates.value(Dates.unix2datetime(time())- start))/1000;
print("TOTAL ELAPSED TIME: ", finish, " seconds. \n")
