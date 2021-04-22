from math import inf
import random
from typing import Tuple, Iterable, Dict, List
from os import path

from matplotlib import pyplot as plt

from knn import input_sample, euclidean_distance
import colors


MAX_FLOAT_LENGTH = 5
DEFAULT_EPSILON = 1 / (10 ** MAX_FLOAT_LENGTH)


def get_mean_for_objects(dots: List[Tuple[float]]) -> Tuple[float]:
    result: Dict[int, float] = {}
    for dot in dots:
        for i in range(len(dot)):
            result[i] = (result.get(i) or 0.0) + dot[i]
    return tuple((result[i] / len(dots) for i in sorted(result.keys())))


def find_clusters_for_objects(
        cluster_centers: List[Tuple[float]], dots: Iterable[Tuple[float]]
) -> Dict[Tuple[float], Dict[Tuple[float], float]]:  # {center: {dot: distance_to_center}}
    result = {}
    for dot in dots:
        closest_center = None
        distance = inf
        for center in cluster_centers:
            current_distance = euclidean_distance(dot, center)
            if current_distance < distance:
                closest_center = center
                distance = current_distance
        if result.get(closest_center):
            result[closest_center][dot] = distance
        else:
            result[closest_center] = {dot: distance}
    result.update({absent_center: {} for absent_center in cluster_centers if absent_center not in result})
    return result


def calculate_max_delta(dot_pairs: Dict[Tuple[float], Tuple[float]]) -> float:
    return max(euclidean_distance(dot_pairs[i], i) for i in dot_pairs)


def normalize_parameters(dots: List[Tuple[float]]) -> Tuple[List[Tuple[float]], float, float]:
    # TODO rewrite (works incorrectly)
    min_coordinate: float = inf
    max_coordinate: float = -inf
    for dot in dots:
        for coordinate in dot:
            min_coordinate = min(min_coordinate, coordinate)
            max_coordinate = max(max_coordinate, coordinate)
    diff = max_coordinate - min_coordinate
    return [tuple((coordinate - min_coordinate) / diff for coordinate in dot) for dot in dots], min_coordinate, max_coordinate


def find_centers_smart(
        dots: List[Tuple[float]], centers_count: int, epsilon: int = DEFAULT_EPSILON, filter_dots: bool = False
) -> List[Tuple[float]]:
    if centers_count < 2:
        raise ValueError("centers_count must be >= 2")
    if filter_dots:
        filtered_dots = []
        for dot in dots:
            skip_dot = False
            for appended_dot in filtered_dots:
                if euclidean_distance(appended_dot, dot) < epsilon:
                    skip_dot = True
                    break
            if not skip_dot:
                filtered_dots.append(dot)
    else:
        filtered_dots = dots

    centers = [random.choice(filtered_dots), ]
    while len(centers) < centers_count:
        clusters = find_clusters_for_objects(centers, filtered_dots)
        distances_to_use: Dict[float, List[Tuple[float]]] = {}
        max_distance = inf
        min_distance = -inf
        for center in clusters:
            for point in clusters[center]:
                if point not in centers:
                    max_distance = max(clusters[center][point], max_distance)
                    min_distance = min(clusters[center][point], min_distance)
                    distance = clusters[center][point]
                    if distances_to_use.get(distance):
                        distances_to_use[distance].append(point)
                    else:
                        distances_to_use[distance] = [point, ]
        random_point = random.uniform(min_distance ** 2, max_distance ** 2)
        centers.append(random.choice(distances_to_use[min(
            distances_to_use.keys(), key=lambda dist: abs((dist ** 2) - random_point)
        )]))
    return centers


def recalculate_centers(
        cluster_centers: List[Tuple[float]], dots: Iterable[Tuple[float]],
        max_iterations: int = 256, epsilon: float = DEFAULT_EPSILON
) -> Dict[Tuple[float], Dict[Tuple[float], float]]:  # {center: {dot: distance_to_center}}
    clusters = find_clusters_for_objects(cluster_centers, dots)
    for _ in range(max_iterations):
        old_centers = list(clusters.keys())
        new_centers = [get_mean_for_objects(list(clusters[center].keys())) for center in clusters]
        if max((euclidean_distance(new_centers[i], old_centers[i]) for i in range(len(new_centers)))) < epsilon:
            break
        clusters = find_clusters_for_objects(new_centers, dots)
    return clusters


def plot_single_object(plot, color, item) -> None:
    if len(item) == 2:
        plot.scatter(*item, color=color)
    else:
        plt.plot(item, color=color)


def main() -> Dict[Tuple[float], Dict[Tuple[float], float]]:
    centers_selection = {
        '1': lambda dots, count: random.sample(dots, count),
        '2': lambda dots, count: dots[:count],
        '3': find_centers_smart
    }
    print("Откуда читать данные (имя файла)?")
    while not (path.exists(input_file_name := input()) and not path.isdir(input_file_name)):
        print("Введено некорректное значение, повторите ввод!")
    input_file = None
    try:
        input_file = open(input_file_name)
        sample = input_sample(input_class=False, file=input_file)
    finally:
        if input_file:
            input_file.close()

    all_dots = [i.values for i in sample]
    for dot in all_dots:
        plot_single_object(plt, 'b', dot)
    plt.show()  # without colors
    plt.clf()
    all_dots, min_parameter, max_parameter = normalize_parameters(all_dots)
    diff = max_parameter - min_parameter

    print(
        "Выберите способ начального выбора центров (только число)\n"
        "  [1] - Случайным образом\n"
        "  [2] - Первые k записей\n"
        "  [3] - k-means++"
    )
    try:
        selection_method = centers_selection[input()]
    except KeyError:
        selection_method = find_centers_smart
        print("Введено некорректное число, будет использовано k-means++")

    while True:
        try:
            k = int(input("Введите К (кол-во кластеров): "))
            if k < 2:
                print(colors.red("Количество должно быть больше чем 2"))
            else:
                break
        except ValueError:
            print(colors.red("Нужно ввести натуральное число!"))

    initial_centers = selection_method(all_dots, k)
    clusters = recalculate_centers(initial_centers, all_dots)
    clusters = {tuple(
        round((param * diff) + min_parameter, MAX_FLOAT_LENGTH) for param in cluster
    ): clusters[cluster] for cluster in clusters}
    for cluster in clusters:
        clusters[cluster] = {tuple(
            round((param * diff) + min_parameter, MAX_FLOAT_LENGTH) for param in dot
        ): clusters[cluster][dot] * diff for dot in clusters[cluster]}
    all_dots = [tuple(round((param * diff) + min_parameter, MAX_FLOAT_LENGTH) for param in dot) for dot in all_dots]

    centers = list(clusters.keys())
    plot_colors = ['b', 'g', 'r', 'y', 'm', 'c', 'k']
    printing_colors = [colors.blue, colors.green, colors.red, colors.yellow, colors.magenta, colors.cyan, colors.grey]

    for i in range(len(centers)):
        for line in clusters[centers[i]]:
            plot_single_object(plt, plot_colors[i % len(plot_colors)], line)
    plt.show()  # colored
    plt.clf()

    max_number_length = max(max(len(coordinate.__repr__()) for coordinate in dot) for dot in all_dots)
    print()
    for dot in all_dots:
        for center in range(len(centers)):
            if min(euclidean_distance(i, dot) for i in clusters[centers[center]]) < DEFAULT_EPSILON:
                for coordinate in dot:
                    print(printing_colors[center](
                        f"{coordinate.__repr__()}{' ' * (max_number_length - len(coordinate.__repr__()))}  "
                    ), end='')
                print(printing_colors[center](f"class: {center}"))
                break
    return clusters


if __name__ == "__main__":
    try:
        main()
    except (EOFError, KeyboardInterrupt):
        print(colors.yellow("\nExit"))
        exit()
