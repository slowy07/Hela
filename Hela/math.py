import math
import Hela.util.error as error

def tambah(a: int, b: int) -> int:
    """
    membuat fungsi pertambahan

    parameter:
        a (int): angka pertama
        b (int): angka kedua
    
    return:
        (int): hasil dari angka pertama + angka kedua
    """
    if isinstance(a, int) and isinstance(b, int):
        return a + b
    else:
        return error.TipeError(["int"])

def kurang(a: int, b: int) -> int:
    """
    membuat fungsi pengurangan

    parameter:
        a (int): angka pertama
        b (int): angka kedua
    
    return:
        (int): hasil dari angka pertama - angka kedua
    """
    if isinstance(a, int) and isinstance(b, int):
        return a - b
    else:
        return error.TipeError(["int"])

def bagi(a: int, b: int) -> int:
    """
    membuat fungsi pembagian

    parameter:
        a (int): angka pertama
        b (int): angka kedua
    
    return:
        (int): hasil dari angka pertama / angka kedua
    """
    if isinstance(a, int) and isinstance(b, int): 
        try:
            return a / b
        except ZeroDivisionError:
            return error.ErrorDibagiNol()
    else:
        return error.TipeError(["int"])

def faktorial(angka: int) -> int:
    """
    menghitung faktorial dari sebuah input nilai
    
    parameter:
        angka (int): angka yang ingin dimasukkan
    return:
        int: angka yang dihasilkan daripada proses faktorial
    """
    if isinstance(angka, int): 
        hasil = 1
        for i in range(1, angka + 1):
            hasil *= i
        return hasil
    else:
        return error.TipeError(["int"])

def jumlah_deret_geometri(utama: int, rasio_umum: int, jumlah: int) -> int:
    """
    fungsi menghitung jumlah suku-suku dari suatu deret geometri

    parameter:
        utama (int): suku utama
        rasio_umum (int): rasio
        jumlah (int): jumlah suku ke n

    return:
        (int): hasil deret geometri dengan rumus Sn = a(1-r^n)/(1-r)
    """
    if isinstance(utama, int) and isinstance(rasio_umum, int) and isinstance(jumlah, int):
        if rasio_umum == 1:
            return jumlah * utama
        return (utama / (1 - rasio_umum)) * (1 - rasio_umum**jumlah)
    else:
        return error.TipeError(["int"])

def modus(arr: list[int | float]) -> int | float:
    """
    fungsi untuk mencari nilai yang sering muncul
    dalam suatu kumpulan data

    parameter:
        arr (list[int atau float]): parameter data

    
    return:
        (int atau float): output ini muncul sebagai sesuai dengan item yang sering terjadi
    """
    if isinstance(arr, list):
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
    else:
        return error.TipeError(["int", "float"])

def normal_pdf(x: float, mean: float, sigma: float) -> float:
    """
    fungsi menghitung kepadatan probabilitas (PDF) dari distribusi normal.

    parameter:
        x (float): nilai input pdf
        mean (float): rata-rata dari distribusi normal
        sigma (float): standar deviasi dari distribusi normal

    return:
        (float): nilai pdf pada input yang diberikan
    """
    if isinstance(x, float) and isinstance(mean, float) and isinstance(sigma, float):
        coefficient = 1 / (sigma * math.sqrt(2 * math.pi))
        exponent = -((x - mean) ** 2) / (2 * sigma**2)
        pdf_value = coefficient * math.exp(exponent)
        return pdf_value
    else:
        return error.TipeError(["int"])