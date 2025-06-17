#!/usr/bin/python

outf = open("test.dat", "w")
for i in range(0, 8191):
    num = "%d " % i
    for j in range(0, 64):
        num += "0.1,"
    num = num.strip(",")
    outf.write(num+"\n")

outf.close();
