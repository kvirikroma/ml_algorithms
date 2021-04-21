from math import inf
import random
from typing import Tuple, Iterable, Dict, List, Set

from knn import SampleItem, input_sample, euclidean_distance, green, red, yellow, blue


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
    return result


def calculate_max_delta(dot_pairs: Dict[Tuple[float], Tuple[float]]) -> float:
    return max(euclidean_distance(dot_pairs[i], i) for i in dot_pairs)


def find_centers_smart(dots: List[Tuple[float]], centers_count: int, epsilon: float = 0.00001) -> List[Tuple[float]]:
    if centers_count < 2:
        raise ValueError("centers_count must be >= 2")
    centers = [random.choice(dots), ]
    while len(centers) < centers_count:
        clusters = find_clusters_for_objects(centers, dots)
        distances_to_use: Dict[float, Set[Tuple[float]]] = {}
        max_distance = inf
        min_distance = 0.0
        for center in clusters:
            for point in clusters[center]:
                for center_to_compare_with in clusters:
                    if euclidean_distance(center_to_compare_with, point) >= epsilon:
                        max_distance = max(clusters[center][point], max_distance)
                        min_distance = min(clusters[center][point], min_distance)
                        distance = clusters[center][point]
                        if distances_to_use.get(distance):
                            distances_to_use[distance].add(point)
                        else:
                            distances_to_use[distance] = {point, }
        random_point = random.uniform(min_distance ** 2, max_distance ** 2)
        centers.append(random.choice(list(distances_to_use[min(
            distances_to_use, key=lambda dist: abs(dist - random_point)
        )])))
    return centers


def recalculate_centers(
        cluster_centers: List[Tuple[float]], dots: Iterable[Tuple[float]], max_iterations: int = 256, epsilon: float = 0.00001
) -> Dict[Tuple[float], Dict[Tuple[float], float]]:  # {center: {dot: distance_to_center}}
    clusters = find_clusters_for_objects(cluster_centers, dots)
    for _ in range(max_iterations):
        old_centers = list(clusters.keys())
        new_centers = [get_mean_for_objects(list(clusters[center].keys())) for center in clusters]
        if min((euclidean_distance(new_centers[i], old_centers[i]) for i in range(len(new_centers)))) < epsilon:
            break
        clusters = find_clusters_for_objects(new_centers, dots)
    return clusters


def main() -> None:
    while True:
        try:
            k = int(input("Введите К (кол-во кластеров): "))
            if k < 2:
                print(red("Количество должно быть больше чем 2"))
            else:
                break
        except ValueError:
            print(red("Нужно ввести натуральное число!"))
    print("Введите выборку элементов:")
    sample = input_sample()


if __name__ == "__main__":
    try:
        main()
    except (EOFError, KeyboardInterrupt):
        print(yellow("\nExit"))
        exit()
