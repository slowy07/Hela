angka: int = 10

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