from typing import Dict, Callable, Optional, Tuple
from multiprocessing import cpu_count
from concurrent.futures import ProcessPoolExecutor
from pandas import DataFrame, read_csv, get_dummies
from sklearn.neighbors import KNeighborsClassifier
from matplotlib import pyplot as plt
import numpy
from sklearn import preprocessing
import json

from colors import yellow, green, blue


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
        classification_algorithm: Callable[[DataFrame, list, DataFrame, list], float], max_steps: int = None
) -> Dict[frozenset, float]:
    result: Dict[str, float] = {}
    result_history: Dict[frozenset, float] = {}
    columns = list(training_data_x.columns)
    executor = ProcessPoolExecutor(max_workers=max(cpu_count() // 2, 1))
    while len(result.keys()) < (
            min(len(training_data_x.columns), max_steps)
            if max_steps is not None
            else len(training_data_x.columns)
    ):
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
        result_history[frozenset(result.keys())] = maximum_accuracy
        print()
        # print(f"{yellow('Rating')}: {rating}")
        print(blue(f"Calculated {len(result)} attributes of {len(training_data_x.columns)}"))
        print(f"{green('Current result')}: {result}")
    executor.shutdown()
    return result_history


def prepare_data_part(data: DataFrame):
    tmp_data = data.copy(deep=True)
    numeric_data = tmp_data.select_dtypes(include=('int16', 'int32', 'int64', 'float16', 'float32', 'float64'))
    non_numeric_data = frozenset(tmp_data.columns) - frozenset(numeric_data)
    non_numeric_data = {column: list(set(tmp_data[column].values)) for column in non_numeric_data}
    tmp_data = get_dummies(tmp_data, columns=non_numeric_data, drop_first=True)
    return tmp_data.reindex(sorted(tmp_data.columns), axis=1)


def prepare_data(data: DataFrame) -> DataFrame:
    values_scaled = preprocessing.MinMaxScaler().fit_transform(data.values)
    final_data = DataFrame(values_scaled)
    final_data.columns = [str(column).strip() for column in final_data.columns]
    return final_data


def main():
    training_data = read_csv(input("Введите расположение файла с данными для обучения: "))
    testing_data = read_csv(input("Введите расположение файла с тестовыми данными: "))
    print()
    for data in training_data, testing_data:
        data.replace(' ?', numpy.nan, inplace=True)
        data.dropna("index", inplace=True, how='any')
        data.columns = [column.strip() for column in data.columns]

    training_data_y_name = list(training_data.columns)[-1]
    training_data_y = list(training_data[training_data_y_name])
    training_data = training_data.drop(training_data_y_name, axis=1)
    testing_data_y_name = list(testing_data.columns)[-1]
    testing_data_y = list(testing_data[testing_data_y_name])
    testing_data = testing_data.drop(testing_data_y_name, axis=1)

    all_values = list(
        frozenset(str(item).strip('. ') for item in training_data_y).union(str(item).strip('. ') for item in testing_data_y)
    )
    all_values_dict = {all_values[i]: i for i in range(len(all_values))}
    for data_y in training_data_y, testing_data_y:
        for i in range(len(data_y)):
            data_y[i] = all_values_dict[str(data_y[i]).strip('. ')]

    testing_data = prepare_data_part(testing_data)
    training_data = prepare_data_part(training_data)
    training_length = len(training_data)
    all_data = training_data.copy(deep=True)
    all_data = all_data.append(testing_data)
    all_data = prepare_data(all_data)
    all_data.columns = training_data.columns
    all_data.dropna(axis=1, inplace=True)
    training_data = all_data.iloc[:training_length, :]
    testing_data = all_data.iloc[training_length:, :]
    testing_data.columns = all_data.columns
    training_data.columns = all_data.columns

    weights_history = forward_selection(
        training_data, list(training_data_y), testing_data, list(testing_data_y), adapter
    )
    plt.figure(figsize=(25, 8))
    sorted_columns = [*sorted(weights_history.keys(), key=lambda x: -len(x), reverse=True)]
    columns_to_draw = [*range(len(sorted_columns))]
    values_to_draw = [weights_history[i] for i in sorted_columns]
    plt.xticks(range(len(columns_to_draw)), columns_to_draw)
    with open("./data/selection_result.json", 'w') as file:
        json.dump({"data": [{"attributes": list(i), "accuracy": weights_history[i]} for i in sorted_columns]}, file, indent=2)
    plt.plot(columns_to_draw, values_to_draw)
    plt.show()


if __name__ == "__main__":
    try:
        main()
        print()
    except (EOFError, KeyboardInterrupt):
        print("\nExit")
