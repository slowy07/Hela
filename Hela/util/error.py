class TipeError(TypeError):
    def __init__(self, tipe_data: list):
        pesan = f"Error: kamu memasukkan tipe data yang salah, seharusnya adalah {' atau '.join(tipe_data)}"
        super().__init__(pesan)

class ErrorDibagiNol(ZeroDivisionError):
    def __init__(self):
        pesan = "Error: tidak bisa dibagikan dengan nol!"
        super().__init__(pesan)