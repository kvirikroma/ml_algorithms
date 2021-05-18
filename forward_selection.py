from typing import Dict, Callable, Optional, Tuple, Union
from multiprocessing import cpu_count
from concurrent.futures import ProcessPoolExecutor
from pandas import DataFrame, read_csv, get_dummies
from sklearn.neighbors import KNeighborsClassifier
from matplotlib import pyplot as plt
import numpy
from sklearn import preprocessing


def adapter(training_data_x: DataFrame, training_data_y: list, test_data_x: DataFrame, test_data_y: list) -> float:
    classifier = KNeighborsClassifier(n_neighbors=3, n_jobs=max(cpu_count() // 2, 1))
    classifier.fit(training_data_x, training_data_y)
    predicted = classifier.predict(test_data_x)
    total_correct = 0
    test_items_count = max(len(test_data_y), len(predicted))
    for i in range(test_items_count):
        total_correct += int(predicted[i] == test_data_y[i])
    return total_correct / test_items_count


def process_func(
        result: dict, column: str, classification_algorithm: Callable[[DataFrame, list, DataFrame, list], float],
        training_data_x, training_data_y, test_data_x, test_data_y
) -> Optional[Tuple[float, str]]:
    if column in result:
        return
    columns = [column, *result.keys()]
    try:
        return classification_algorithm(
            training_data_x[columns], training_data_y, test_data_x[columns], test_data_y
        ), column
    except KeyError:
        return


def forward_selection(
        training_data_x: DataFrame, training_data_y: list, test_data_x: DataFrame, test_data_y: list,
        classification_algorithm: Callable[[DataFrame, list, DataFrame, list], float]
) -> Dict[str, float]:
    result: Dict[str, float] = {}
    columns = list(training_data_x.columns)
    executor = ProcessPoolExecutor(max_workers=max(cpu_count() // 2, 1))
    while len(result.keys()) < len(training_data_x.columns):
        mapped_rating = executor.map(
            process_func,
            (result for _ in columns),
            columns,
            (classification_algorithm for _ in columns),
            (training_data_x for _ in columns),
            (training_data_y for _ in columns),
            (test_data_x for _ in columns),
            (test_data_y for _ in columns)
        )
        rating: Dict[float, str] = {
            accuracy: column for accuracy, column in (item for item in mapped_rating if item)
        }
        if not rating:
            break
        maximum_accuracy = max(rating.keys())
        result[rating[maximum_accuracy]] = maximum_accuracy
        print()
        print(f"Calculated {len(result)} attributes of {len(training_data_x.columns)}")
        print(result)
    executor.shutdown()
    return result


def prepare_data(data: DataFrame) -> DataFrame:
    tmp_data = data.dropna("index", how='any')
    numeric_data = tmp_data.select_dtypes(include=('int16', 'int32', 'int64', 'float16', 'float32', 'float64'))
    non_numeric_data = frozenset(tmp_data.columns) - frozenset(numeric_data)
    non_numeric_data = {column: list(set(tmp_data[column].values)) for column in non_numeric_data}
    tmp_data = get_dummies(tmp_data, columns=non_numeric_data, drop_first=True)
    values_scaled = preprocessing.MinMaxScaler().fit_transform(tmp_data.values)
    final_data = DataFrame(values_scaled)
    final_data.columns = [column.strip() for column in tmp_data.columns]
    # final_data.columns = [str(i) for i in range(len(columns))]
    return tmp_data


def main():
    training_data = read_csv(input("Введите расположение файла с данными для обучения: "))
    testing_data = read_csv(input("Введите расположение файла с тестовыми данными: "))
    for data in training_data, testing_data:
        data.columns = [column.strip() for column in data.columns]
        data.replace(' ?', numpy.nan, inplace=True)
        data.dropna("index", inplace=True, how='any')
    training_length = len(training_data)
    all_data = training_data.copy(deep=True)
    all_data = all_data.append(testing_data)
    all_data = prepare_data(all_data)
    training_data = all_data.iloc[:training_length, :]
    testing_data = all_data.iloc[training_length:, :]
    training_data_y_name = list(training_data.columns)[-1]
    training_data_y = training_data[training_data_y_name]
    training_data = training_data.drop(training_data_y_name, axis=1)
    testing_data_y_name = list(testing_data.columns)[-1]
    testing_data_y = testing_data[testing_data_y_name]
    testing_data = testing_data.drop(testing_data_y_name, axis=1)
    weights = forward_selection(training_data, list(training_data_y), testing_data, list(testing_data_y), adapter)
    print(f"\nResult: {weights}")
    plt.figure(figsize=(25, 8))
    sorted_columns = [*sorted(weights.keys(), key=lambda x: weights[x], reverse=True)]
    columns_to_draw = [
        f'{sorted_columns[i]}_{i}' if
        sorted_columns[i].count('_') == 0 else
        f'{sorted_columns[i][0]}{sorted_columns[i].split("_")[1].strip()[:1]}_{i}'
        for i in range(len(sorted_columns))
    ]
    values_to_draw = [*sorted(weights.values(), reverse=True)]
    plt.xticks(range(len(columns_to_draw)), columns_to_draw)
    plt.plot(columns_to_draw, values_to_draw)
    plt.show()


if __name__ == "__main__":
    try:
        main()
        print()
    except (EOFError, KeyboardInterrupt):
        print("\nExit")
