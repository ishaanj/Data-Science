def isKap(x):
    if x <= 3: return x == 1
    y = str(x*x)
    d = len(str(x))
    a = int(y[-d:])
    b = int(y[:-d])
    if x == a+b: return x

p = int(input())
q = int(input())
l = []

for i in range(p,q+1):
    if isKap(i): l.append(i)

for x in l:
    if x%10 == 0: l.remove(x)

if len(l) == 0:
    print("Invalid Range")
else:
    print(*l, sep=" ")