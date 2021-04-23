from math import inf
import random
from typing import Tuple, Iterable, Dict, List
from os import path

from matplotlib import pyplot as plt

from knn import input_sample, euclidean_distance
import colors


MAX_FLOAT_LENGTH = 5
DEFAULT_EPSILON = 1 / (10 ** MAX_FLOAT_LENGTH)
RETURN_NON_NORMALIZED = True


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


def normalize_parameters(dots: List[Tuple[float]]) -> Tuple[List[Tuple[float]], List[Tuple[float, float]]]:
    min_dot_len = min(len(dot) for dot in dots)
    minimums_and_maximums = [(inf, -inf, ) for _ in range(min_dot_len)]
    for coordinate in range(min_dot_len):
        for i in range(len(dots)):
            minimums_and_maximums[coordinate] = (
                min(minimums_and_maximums[coordinate][0], dots[i][coordinate]),
                max(minimums_and_maximums[coordinate][1], dots[i][coordinate]),
            )
    diffs = [min_and_max[1] - min_and_max[0] for min_and_max in minimums_and_maximums]
    result = [tuple((float(dot[i]) - minimums_and_maximums[i][0]) / diffs[i] for i in range(min_dot_len)) for dot in dots]
    return result, minimums_and_maximums


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
        dots_to_distances = {}
        max_distance = inf
        min_distance = -inf
        for center in clusters:
            for point in clusters[center]:
                if point not in centers:
                    max_distance = max(clusters[center][point], max_distance)
                    min_distance = min(clusters[center][point], min_distance)
                    distance = clusters[center][point]
                    dots_to_distances[point] = distance
        all_keys = list(dots_to_distances.keys())
        centers.append(random.choices(all_keys, weights=[dots_to_distances[dot] for dot in all_keys], k=1)[0])
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

    all_dots: List[Tuple[float]] = [i.values for i in sample]
    for dot in all_dots:
        plot_single_object(plt, 'b', dot)
    plt.show()  # without colors
    plt.clf()
    normalized_dots, minimums_and_maximums = normalize_parameters(all_dots)
    diffs = [min_and_max[1] - min_and_max[0] for min_and_max in minimums_and_maximums]

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

    initial_centers = selection_method(normalized_dots, k)
    clusters = recalculate_centers(initial_centers, normalized_dots)

    if RETURN_NON_NORMALIZED:
        clusters = {tuple(round(
            (cluster[i] * diffs[i]) + minimums_and_maximums[i][0], MAX_FLOAT_LENGTH
        ) for i in range(len(cluster))): clusters[cluster] for cluster in clusters}
        for cluster in clusters:
            clusters[cluster] = {tuple(
                round((dot[i] * diffs[i]) + minimums_and_maximums[i][0], MAX_FLOAT_LENGTH) for i in range(len(dot))
            ): clusters[cluster][dot] for dot in clusters[cluster]}

    centers = list(clusters.keys())
    plot_colors = ['b', 'g', 'r', 'y', 'm', 'c', 'k']
    printing_colors = [colors.blue, colors.green, colors.red, colors.yellow, colors.magenta, colors.cyan, colors.grey]

    for i in range(len(centers)):
        for line in clusters[centers[i]]:
            plot_single_object(plt, plot_colors[i % len(plot_colors)], line)
    plt.show()  # colored
    plt.clf()

    max_number_length = max(max(len(coordinate.__repr__()) for coordinate in dot) for dot in (
        all_dots if RETURN_NON_NORMALIZED else normalized_dots
    ))
    print()
    for dot in (all_dots if RETURN_NON_NORMALIZED else normalized_dots):
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
