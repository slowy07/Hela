def tambah(a: int, b: int) -> int:
    return a + b

def kurang(a: int, b: int) -> int:
    return a - b

def bagi(a: int, b: int) -> int:
    return a / b

def faktorial(angka: int) -> int:
    hasil = 1
    for i in range(1, angka + 1):
        hasil *= i
    return hasil

def jumlah_deret_geometri(utama: int, rasio_umum: int, jumlah: int) -> int:
    if rasio_umum == 1:
        return jumlah * utama
    return (utama / (1 - rasio_umum)) * (1 - rasio_umum**jumlah)

def modus(arr: list[int | float]) -> int | float:
    count = []
    for value in arr:
        count.append(arr.count(value))
    combine = dict(zip(arr, count))
    memo = [a for a, b in combine.items() if b == max(count)]
    min_value = memo[0]
    result = None
    for value in memo:
        if value < min_value:
            min_value = value
    result = min_value
    if result == 1:
        raise ValueError("nan")
    return result
