from math import inf
import random
from typing import Iterable, Dict, List, TextIO
from os import path

from matplotlib import pyplot as plt

import colors

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

    def min_distance(self, other: 'Cluster') -> float:
        try:
            return min(self_dot.distance(other_dot) for self_dot in self for other_dot in other)
        except ValueError:
            return inf

    def max_self_distance(self) -> float:
        try:
            return max(dot.distance(other_dot) for dot in self for other_dot in self if dot != other_dot)
        except ValueError:
            return 0.0


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
        plot.plot(item, color=color)


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


def dunn_index(clusters: List[Cluster]):
    max_in_cluster_distance = max(cluster.max_self_distance() for cluster in clusters)
    min_clusters_distance = min(
        cluster.min_distance(other_cluster)
        for cluster in clusters
        for other_cluster in clusters
        if cluster != other_cluster
    )
    return min_clusters_distance / max_in_cluster_distance


def main():
    centers_selection = {
        '1': lambda dots, count: create_clusters(random.sample(dots, count), dots),
        '2': lambda dots, count: create_clusters(dots[:count], dots),
        '3': k_means_plusplus
    }
    while not (
            path.exists(input_file_name := input("Enter the name of file to read the data from: "))
            and not path.isdir(input_file_name)
    ):
        print(colors.red("Invalid input"))
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
        "Choose the way of initial centers choosing (input only the number)\n"
        "  [1] - Randomly\n"
        "  [2] - First k samples\n"
        "  [3] - k-means++"
    )
    try:
        selection_method = centers_selection[input()]
    except KeyError:
        selection_method = k_means_plusplus
        print("Incorrect input. K-means++ will be used.")

    while True:
        try:
            k = int(input("Input К (number of clusters): "))
            if k < 2:
                print("Number of clusters must be more than 2")
            else:
                break
        except ValueError:
            print("The number must be natural")

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

    quality_index = dunn_index(clusters)
    print(f"Dunn index for this clustering: {quality_index}")

    plt.clf()
    for i in range(len(centers)):
        for item in clusters[i]:
            display_on_plot(plt, plot_colors[i % len(plot_colors)], item)
    plt.figure()
    plt.show(block=False)  # colored

    clustering_qualities = {}
    for i in range(2, 17):
        if i == k:
            clustering_qualities[i] = quality_index
        else:
            tmp_init_clusters = selection_method(all_dots, i)
            tmp_clusters = make_centers_mean([cluster.center for cluster in tmp_init_clusters], all_dots)
            clustering_qualities[i] = dunn_index(tmp_clusters)
    print("Dunn index for other clustering types:")
    for i in clustering_qualities:
        print(f"{i}: {clustering_qualities[i]}")
    plt.clf()
    plt.plot([None, None, *(clustering_qualities[clusters_count] for clusters_count in sorted(clustering_qualities))])
    plt.show()


if __name__ == "__main__":
    try:
        main()
    except (EOFError, KeyboardInterrupt):
        print(colors.yellow("\nExit"))
        exit()
