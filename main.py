import matplotlib.pyplot as plt
import numpy as np
from scipy.ndimage import shift
from typing import Iterable


def read_csv_dataset() -> list[str]:
    with open("dataset.csv", "r") as ds:
        return ds.read().split()


def process_year(year_dataset: Iterable[list[str]]) -> None:
    plt.figure(figsize=(10, 10))
    for row in year_dataset:
        row_name = row[0]
        x_k = np.arange(0.1, 1.1, 0.1)
        x_k_1 = shift(x_k, 1, cval=0)

        income_deciles = [int(v) for v in row[1:]]
        income_k_cum = np.cumsum(income_deciles)
        cum_income = income_k_cum/income_k_cum[-1]
        cum_income_k_1 = shift(cum_income, 1, cval=0)

        square_1 = (x_k_1 + x_k) / 2 * 0.1
        sum_square_1 = sum(square_1)
        square_2 = (cum_income_k_1 + cum_income) / 2 * 0.1
        sum_square_2 = sum(square_2)
        diff = sum_square_1 - sum_square_2
        gini = diff / sum_square_1

        print(f"Gini for {row_name}: {gini}")
        indices = np.linspace(0, len(cum_income) - 1, len(x_k))
        interp_cum_income = np.interp(indices, np.arange(len(cum_income)), cum_income)
        plt.plot(
            np.insert(x_k, 0, 0),
            np.insert(interp_cum_income, 0, 0),
            label=f'{row_name} (Gini = {round(gini, 2)})'
        )

    plt.plot(([0, 0], [1, 1]))
    plt.title('Кривая Лоренца')
    plt.xlabel('Кумулятивная доля населения')
    plt.ylabel('Кумулятивная доля дохода')
    plt.legend(loc="upper left")
    plt.xlim(0, 1)
    plt.ylim(0, 1)
    plt.show()


def process_dataset(dataset: list[str]) -> None:
    for idx in range(0, len(dataset), 4):
        year_specific_slice = dataset[idx:idx + 4]
        splitted_slice = (row.strip().split(",") for row in year_specific_slice)
        process_year(splitted_slice)


def run() -> None:
    dataset = read_csv_dataset()
    process_dataset(dataset)


if __name__ == '__main__':
    run()
