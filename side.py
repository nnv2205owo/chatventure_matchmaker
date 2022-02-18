a = list(map(int, input().split()))
b, c = map(int, input().split())

a = a[:b - 1] + [c] + a[b - 1:]
print(a)
