from math import inf
import random
from typing import Iterable, Dict, List, TextIO
from os import path

from matplotlib import pyplot as plt


MAX_FLOAT_LENGTH = 6
DEFAULT_EPSILON = 1 / (10 ** MAX_FLOAT_LENGTH)


class Dot:
    def __init__(self, coordinates: Iterable[float]):
        self._coordinates = tuple(coordinates)

    def __len__(self):
        return len(self._coordinates)

    def __iter__(self):
        return iter(self._coordinates)

    def __getitem__(self, item) -> float:
        return self._coordinates[item]

    def __hash__(self):
        return hash(self._coordinates)

    def __eq__(self, other: 'Dot') -> bool:
        if len(self) != len(other):
            return False
        return self.distance(other) < DEFAULT_EPSILON

    def distance(self, other: 'Dot') -> float:
        result = 0
        for coord in range(min(len(self), len(other))):
            result += (self[coord] - other[coord]) ** 2
        return result ** 0.5


class Cluster:
    def __init__(self, center: Dot, dots: Dict[Dot, float]):
        self.center = center
        self.dots = {**dots}

    @property
    def mean(self) -> Dot:
        result: Dict[int, float] = {}
        for dot in self.dots:
            for i in range(len(dot)):
                result[i] = (result.get(i) or 0.0) + dot[i]
        return Dot(result[i] / len(self.dots) for i in sorted(result))

    def __contains__(self, item: Dot) -> bool:
        return min(item.distance(dot) for dot in self.dots) < DEFAULT_EPSILON

    def __iter__(self):
        return iter(self.dots)

    def __getitem__(self, item: Dot) -> float:
        return self.dots[item]


def create_clusters(cluster_centers: List[Dot], dots: Iterable[Dot]) -> List[Cluster]:
    result: Dict[Dot, Dict[Dot, float]] = {}
    for dot in dots:
        closest_center = None
        distance = inf
        for center in cluster_centers:
            current_distance = dot.distance(center)
            if current_distance < distance:
                closest_center = center
                distance = current_distance
        if result.get(closest_center):
            result[closest_center][dot] = distance
        else:
            result[closest_center] = {dot: distance}
    result.update({absent_center: {} for absent_center in cluster_centers if absent_center not in result})
    return [Cluster(i, result[i]) for i in result]


def normalize_parameters(dots: List[Dot]) -> List[Dot]:
    min_dot_len = min(len(dot) for dot in dots)
    minimums_and_maximums = [(inf, -inf, ) for _ in range(min_dot_len)]
    for coordinate in range(min_dot_len):
        for i in range(len(dots)):
            minimums_and_maximums[coordinate] = (
                min(minimums_and_maximums[coordinate][0], dots[i][coordinate]),
                max(minimums_and_maximums[coordinate][1], dots[i][coordinate]),
            )
    diffs = [min_and_max[1] - min_and_max[0] for min_and_max in minimums_and_maximums]
    result = [Dot((dot[i] - minimums_and_maximums[i][0]) / diffs[i] for i in range(min_dot_len)) for dot in dots]
    return result


def k_means_plusplus(dots: List[Dot], centers_count: int) -> List[Cluster]:
    if centers_count < 2:
        raise ValueError("centers_count must be >= 2")
    centers = [random.choice(dots), ]
    while len(centers) < centers_count:
        clusters = create_clusters(centers, dots)
        dots_to_distances: Dict[Dot, float] = {}
        for cluster in clusters:
            for point in cluster:
                if point not in centers:
                    distance = cluster[point]
                    dots_to_distances[point] = distance
        all_keys = list(dots_to_distances.keys())
        centers.append(random.choices(all_keys, weights=[dots_to_distances[dot] ** 2 for dot in all_keys], k=1)[0])
    return create_clusters(centers, dots)


def make_centers_mean(cluster_centers: List[Dot], dots: Iterable[Dot], max_iterations: int = 256) -> List[Cluster]:
    clusters = create_clusters(cluster_centers, dots)
    for _ in range(max_iterations):
        old_centers = [cluster.center for cluster in clusters]
        new_centers = [cluster.mean for cluster in clusters]
        if max((new_centers[i].distance(old_centers[i]) for i in range(len(new_centers)))) < DEFAULT_EPSILON:
            break
        clusters = create_clusters(new_centers, dots)
    return clusters


def display_on_plot(plot, color, item) -> None:
    if len(item) == 2:
        plot.scatter(*item, color=color)
    else:
        plt.plot(item, color=color)


def input_sample(file: TextIO) -> List[Dot]:
    result = []
    while True:
        new_line = file.readline()
        if new_line:
            new_line = new_line.replace('\r', '').replace('\n', '')
        else:
            break
        numbers_str = [part.replace(',', '.') for part in new_line.split() if part]
        if len(numbers_str) < 2:
            continue
        item_coordinates = []
        line_parsed = True
        for i in range(len(numbers_str)):
            try:
                item_coordinates.append(float(numbers_str[i]))
            except ValueError:
                line_parsed = False
                break
        if not line_parsed:
            continue
        result.append(Dot(item_coordinates))
    return result


def main():
    centers_selection = {
        '1': lambda dots, count: create_clusters(random.sample(dots, count), dots),
        '2': lambda dots, count: create_clusters(dots[:count], dots),
        '3': k_means_plusplus
    }
    print("Откуда читать данные (имя файла)?")
    while not (path.exists(input_file_name := input()) and not path.isdir(input_file_name)):
        print("Введено некорректное значение, повторите ввод!")
    input_file = None
    try:
        input_file = open(input_file_name)
        all_dots = input_sample(file=input_file)
    finally:
        if input_file:
            input_file.close()

    for dot in all_dots:
        display_on_plot(plt, 'b', dot)
    plt.show(block=False)  # without colors
    all_dots = normalize_parameters(all_dots)

    print(
        "Выберите способ начального выбора центров (только число)\n"
        "  [1] - Случайным образом\n"
        "  [2] - Первые k записей\n"
        "  [3] - k-means++"
    )
    try:
        selection_method = centers_selection[input()]
    except KeyError:
        selection_method = k_means_plusplus
        print("Введено некорректное число, будет использовано k-means++")

    while True:
        try:
            k = int(input("Введите К (кол-во кластеров): "))
            if k < 2:
                print("Количество должно быть больше чем 2")
            else:
                break
        except ValueError:
            print("Нужно ввести натуральное число!")

    initial_clusters = selection_method(all_dots, k)
    clusters = make_centers_mean([cluster.center for cluster in initial_clusters], all_dots)

    centers = [cluster.center for cluster in clusters]
    plot_colors = ['b', 'g', 'r', 'y', 'm', 'c', 'k']

    max_number_length = max(max(len(coordinate.__repr__()) for coordinate in dot) for dot in all_dots)
    print()
    for dot in all_dots:
        for cluster in range(len(clusters)):
            if dot in clusters[cluster]:
                for coordinate in dot:
                    print(f"{coordinate.__repr__()}{' ' * (max_number_length - len(coordinate.__repr__()))}  ", end='')
                print(f"class: {cluster}")
                break

    plt.clf()
    for i in range(len(centers)):
        for item in clusters[i]:
            display_on_plot(plt, plot_colors[i % len(plot_colors)], item)
    plt.show()  # colored
    plt.clf()


if __name__ == "__main__":
    try:
        main()
    except (EOFError, KeyboardInterrupt):
        print("\nExit")
        exit()
