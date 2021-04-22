from typing import Tuple, List, Dict, Optional, TextIO
from sys import stdin

from colors import green, red, yellow, blue


class SampleItem:
    def __init__(self, values: Tuple, class_name: Optional[str]):
        self.values = values
        self.class_name = class_name


def euclidean_distance(point1: Tuple[float], point2: Tuple[float]) -> float:
    result = 0
    for coord in range(min(len(point1), len(point2))):
        result += (point1[coord] - point2[coord])**2
    return result ** 0.5


def chebyshev_distance(point1: Tuple[float], point2: Tuple[float]) -> float:
    return max(abs(point1[coord] - point2[coord]) for coord in range(min(len(point1), len(point2))))


def manhattan_distance(point1: Tuple[float], point2: Tuple[float]) -> float:
    return sum(abs(point1[coord] - point2[coord]) for coord in range(min(len(point1), len(point2))))


def knn_predict(training_sample: List[SampleItem], item_values: Tuple[float], distance_calculator, k: int) -> str:
    if k < 1:
        raise ValueError(f"K must be natural ({k} < 1)")
    distances = [
        (
            distance_calculator(training_sample[i].values, item_values),
            training_sample[i].class_name
        ) for i in range(len(training_sample))
    ]
    distances.sort(key=lambda x: x[0])
    while True:
        classes_predicted = [item[1] for item in distances[:k]]
        predicted_dict = {classes_predicted.count(item): item for item in set(classes_predicted)}
        if len(predicted_dict.keys()) == len(set(classes_predicted)):
            break
        k -= 1
    return predicted_dict[max(predicted_dict.keys())]


def print_distances_table(training_sample: List[SampleItem], test_sample: List[SampleItem], distance_calculator, k: int):
    print(end='    \t')
    for i in range(len(training_sample)):
        print(end=yellow(f'tr{i}\t'))
    print()
    i = 0
    for test_item in test_sample:
        print(end=blue(f'te{i}:\t'))
        i += 1
        distances = []
        for training_item in training_sample:
            distances.append(round(distance_calculator(test_item.values, training_item.values), 3))
        coloured_distance = max(sorted(distances)[:k])
        for training_item in distances:
            if training_item <= coloured_distance:
                training_item = green(str(training_item))
            print(training_item, end='\t')
        print()


def input_sample(input_class: bool = True, file: TextIO = stdin) -> List[SampleItem]:
    if file == stdin:
        print(
            "Вводите элементы (новый в каждой строке), вводя их параметры через пробел"
            f"{' и имя класса в конце (также через пробел).' if input_class else ''} "
            "Закончите ввод двумя пустыми строками."
        )
    result = []
    empty_line_entered = False
    while True:
        new_line = file.readline().replace('\r', '').replace('\n', '')
        if new_line:
            empty_line_entered = False
        else:
            if empty_line_entered:
                break
            else:
                empty_line_entered = True
                continue
        numbers_str = [part.replace(',', '.') for part in new_line.split() if part]
        if len(numbers_str) < 2:
            if file == stdin:
                print("Введено неверное значение, оно не будет учтено")
            continue
        item_coordinates = []
        line_parsed = True
        for i in range(len(numbers_str) - int(input_class)):
            try:
                item_coordinates.append(float(numbers_str[i]))
            except ValueError:
                line_parsed = False
                break
        if not line_parsed:
            if file == stdin:
                print("Введено неверное значение, оно не будет учтено")
            continue
        result.append(SampleItem(tuple(item_coordinates), numbers_str[-1] if input_class else None))
    return result


def print_sample(sample: List[SampleItem], ys_predicted: List[str] = None):
    print(end='    \t')
    for i in range(max(len(sample_item.values) for sample_item in sample)):
        print(end=f'X{i}\t')
    print(f'{blue("Y")}\t{yellow("Y_pred")}\tResult' if ys_predicted else blue('Y'))
    i = 0
    for item in sample:
        print(end=f'{i}\t')
        for coordinate in item.values:
            print(end=f'{coordinate}\t')
        print(
            f'{blue(item.class_name)}\t{yellow(ys_predicted[i])}\t '
            f'{green("✔️") if ys_predicted[i] == item.class_name else red("❌")}'
            if ys_predicted else blue(item.class_name)
        )
        i += 1


def get_used_classes(real_classes: List[str], predicted_classes: List[str]) -> List[str]:
    return sorted(frozenset(predicted_classes).union(frozenset(real_classes)))


def calculate_confusion_matrix(real_classes: List[str], predicted_classes: List[str]) -> Dict[str, Dict[str, int]]:
    classes = get_used_classes(real_classes, predicted_classes)
    matrix = {
        predicted_class: {
            real_class: 0 for real_class in classes
        } for predicted_class in classes
    }
    for i in range(min(len(real_classes), len(predicted_classes))):
        matrix[predicted_classes[i]][real_classes[i]] += 1
    return matrix


def print_confusion_matrix(matrix: Dict[str, Dict[str, int]]):
    classes = get_used_classes(
        list(matrix.keys()),
        list(matrix[list(matrix.keys())[0]].keys())
    )
    print(end='      \t')
    for predicted_class in classes:
        print(end=yellow(f'{predicted_class}_pred\t'))
    print()
    for real_class in classes:
        print(end=blue(f'{real_class}_real\t'))
        for predicted_class in classes:
            print(end=(green if predicted_class == real_class else red)(f'{matrix[predicted_class][real_class]}\t'))
        print()


def main():
    distance_choices = {'1': euclidean_distance, '2': manhattan_distance, '3': chebyshev_distance}
    print(
        "Выберите способ подсчёта расстояния (только число)\n"
        "[1] - Евклидов способ\n"
        "[2] - Манхеттенский способ\n"
        "[3] - Способ Чебышева"
    )

    # this disables some stupid PyCharm warnings
    k = training_sample = test_sample = predicted_classes = selected_distance_calc = None

    try:
        selected_distance_calc = distance_choices[input()]
    except KeyError:
        selected_distance_calc = euclidean_distance
        print("Введено некорректное число, будет использовано Евклидово расстояние")
    except (EOFError, KeyboardInterrupt):
        print(yellow("\nExit"))
        exit()

    try:
        k = int(input("Введите К (гиперпараметр класификатора): "))
        print("Введите обучающую выборку:")
        training_sample = input_sample()
        print("Введите тестовую выборку:")
        test_sample = input_sample()
    except (EOFError, KeyboardInterrupt):
        print(yellow("\nExit"))
        exit()

    print("\nТаблица расстояний для объектов:")
    print_distances_table(training_sample, test_sample, selected_distance_calc, k)

    try:
        predicted_classes = [knn_predict(training_sample, item.values, selected_distance_calc, k) for item in test_sample]
    except ValueError as err:
        print('\n' + red(err.args if isinstance(err.args, str) else err.args[0]))
        print(yellow("Exit"))
        exit()

    print("\nТаблица предсказаний класификатора для тестовой выборки:")
    print_sample(test_sample, predicted_classes)
    confusion_matrix = calculate_confusion_matrix([item.class_name for item in test_sample], predicted_classes)
    print("\nМатрица ошибок:")
    print_confusion_matrix(confusion_matrix)

    correct_predicts = sum((1 for i in range(len(predicted_classes)) if predicted_classes[i] == test_sample[i].class_name))
    correctness_percent = correct_predicts / len(predicted_classes)
    correctness_color = red
    if correctness_percent > 0.5:
        correctness_color = yellow
    if correctness_percent > 0.75:
        correctness_color = green
    print(f"\nТочность класификатора: {correctness_color(str(correctness_percent))}")


if __name__ == "__main__":
    main()
