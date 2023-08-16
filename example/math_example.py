from Hela.math import Math


def finding_radians() -> None:
    """
    finding radians
    """
    hela = Math()

    angle_in_radians: int = 90
    print(f"radians in {angle_in_radians} is: {hela.radians(angle_in_radians)}")


def mean_of_list_numbers() -> None:
    """
    finding mean of given number
    """
    hela = Math()

    avg_mean_list: list = [1, 2, 3, 4, 5]
    print(f"average mean of {avg_mean_list} is: {hela.avg_mean(avg_mean_list)}")


def finding_cosine() -> None:
    """
    finding cosine
    """
    hela = Math()
    angle_in_radians: int = 0
    print(
        f"value cosine in {angle_in_radians} angle is {hela.cosine(angle_in_radians)}"
    )


finding_radians()
mean_of_list_numbers()
finding_cosine()
