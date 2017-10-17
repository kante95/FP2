using CSV
using DataFrames
using Gadfly


x = CSV.read("../Invariant_Masses.txt",delim=' ',header=["data", "type"])
print(x)

p = plot(x=x[:data], Geom.histogram)
draw(PNG("histogram.png", 17inch, 10inch), p)