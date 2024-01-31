import unittest
import Hela.math as math
import Hela.util.error as error


class TestingTambah(unittest.TestCase):
    def test_operasi(self):
        aktual = math.tambah(2, 3)
        ekspetasi = 2 + 3
        self.assertEqual(aktual, ekspetasi)

    def test_nilai_salah(self):
        aktual = math.tambah(2, 2.3)
        with self.assertRaises(error.TipeError):
            raise aktual


class TestingKurang(unittest.TestCase):
    def test_operasi(self):
        aktual = math.kurang(20, 5)
        ekspetasi = 20 - 5
        self.assertEqual(aktual, ekspetasi)

    def test_nilai_salah(self):
        aktual = math.kurang("20", 5)
        with self.assertRaises(error.TipeError):
            raise aktual


class TestingBagi(unittest.TestCase):
    def test_operasi(self):
        hasil = math.bagi(100, 25)
        self.assertEqual(hasil, 4)

    def test_nilai_salah(self):
        hasil = math.bagi("100", 25)
        with self.assertRaises(error.TipeError):
            raise hasil

    def test_dibagi_nol(self):
        aktual = math.bagi(8, 0)
        with self.assertRaises(error.ErrorDibagiNol):
            raise aktual


class TestFaktorial(unittest.TestCase):
    def test_operasi(self):
        aktual = math.faktorial(5)
        self.assertEqual(aktual, 120)

    def test_nilai_salah(self):
        aktual = math.faktorial("5")
        with self.assertRaises(error.TipeError):
            raise aktual


class TestJumlahDeretGeometri(unittest.TestCase):
    def test_operasi(self):
        aktual = math.jumlah_deret_geometri(2, 3, 4)
        self.assertEqual(aktual, 80.0)

    def test_nilai_salah(self):
        aktual = math.jumlah_deret_geometri("2", 4, 2)
        with self.assertRaises(error.TipeError):
            raise aktual


class TestModus(unittest.TestCase):
    def test_operasi(self):
        hasil = math.modus([1, 2, 2, 3, 4, 4, 5, 5, 5])
        self.assertEqual(hasil, 5)

    def test_nilai_salah(self):
        hasil = math.modus("2")
        with self.assertRaises(error.TipeError):
            raise hasil


class TestNormalPdf(unittest.TestCase):
    def test_operasi(self):
        hasil = math.normal_pdf(1.2, 1.3, 1.1)
        self.assertEqual(hasil, 0.36117923630158333)
