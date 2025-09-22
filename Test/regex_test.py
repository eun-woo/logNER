import regex, time

data = "abc123xyz"

p1 = regex.compile(r'^[a-z]+\d+[a-z]+$', flags=regex.POSIX)
p2 = r'^[a-z]+\d+[a-z]+$'
p3 = regex.compile(p2)
print(p1.search(data))  # POSIX 모드
print(p3.fullmatch(data))  # 기본 모드

# 성능 비교
import timeit
print("POSIX:", timeit.timeit(lambda: p1.fullmatch(data), number=100000))
print("Default:", timeit.timeit(lambda: p3.fullmatch(data), number=100000))

print(regex.fullmatch(r'^[a-z]+\d+[a-z]+$', data))
print("POSIX:", timeit.timeit(lambda: regex.fullmatch(r'^[a-z]+\d+[a-z]+$', data), number=100000))
print("Default:", timeit.timeit(lambda: regex.fullmatch(r'^[a-z]+\d+[a-z]+$', data), number=100000))