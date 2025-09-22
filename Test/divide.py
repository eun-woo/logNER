for num in range(10000, 1000001, 10000):
    lines = []
    with open('BGL.log') as f1:
        for i in range(num):
            try:
                lines.append(f1.readline())
            except UnicodeDecodeError:
                continue

    filename = f'BGL{num:07d}.log'
    with open(f'./bgl_logs/{filename}', 'w') as f2:
        f2.writelines(lines)

    print(f'{num / 10000}% 작업 완료')