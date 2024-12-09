from __future__ import annotations
from math import sqrt
import numpy as np
from shapely.geometry import Polygon, mapping
from scipy.optimize import minimize
from itertools import combinations
from random import sample
import cv2

class _Line:
    """
    Class to represent a line in 2D space defined by two points (x1, y1) and (x2, y2).
    The line equation is represented in the form Ax + By + C = 0.
    """

    def __init__(self, x1: float | int | None = None, y1: float | int | None = None, x2: float | int | None = None,
                 y2: float | int | None = None,
                 A: float | int | None = None, B: float | int | None = None, C: float | int | None = None):
        """
        Initialize a _Line instance with two points (x1, y1) and (x2, y2).

        :param x1: float | int. x-coordinate of the first point. If None, A, B, and C must be specified.
        :param y1: float | int. y-coordinate of the first point. If None, A, B, and C must be specified.
        :param x2: float | int. x-coordinate of the second point. If None, A, B, and C must be specified.
        :param y2: float | int. y-coordinate of the second point. If None, A, B, and C must be specified.

        :param A: float | int. Coefficient A of the line equation Ax + By + C = 0. If None, x1, y1, x2, and y2 must be specified.
        :param B: float | int. Coefficient B of the line equation Ax + By + C = 0. If None, x1, y1, x2, and y2 must be specified.
        :param C: float | int. Coefficient C of the line equation Ax + By + C = 0. If None, x1, y1, x2, and y2 must be specified.
        """
        if x1 is not None and y1 is not None and x2 is not None and y2 is not None:
            self.A, self.B, self.C = self.calculate_line_coefficients(x1=x1, y1=y1, x2=x2, y2=y2)
        elif A is not None and B is not None and C is not None:
            self.A, self.B, self.C = A, B, C
        else:
            raise ValueError("Either (x1, y1, x2, y2) or (A, B, C) must be specified.")
        self._norm = sqrt(self.A*self.A + self.B * self.B)

    def __copy__(self):
        return _Line(A=self.A, B=self.B, C=self.C)

    def copy(self):
        return self.__copy__()

    def calculate_line_coefficients(self, x1: float | int, y1: float | int, x2: float | int, y2: float | int) -> tuple[
        float, float, float]:
        """
        Calculate the coefficients A, B, and C of the line equation Ax + By + C = 0.

        :param x1: float | int. x-coordinate of the first point.
        :param y1: float | int. y-coordinate of the first point.
        :param x2: float | int. x-coordinate of the second point.
        :param y2: float | int. y-coordinate of the second point.
        :return: tuple[float, float, float]. Tuple containing the coefficients A, B, and C.
        """
        # Equation derived from two-point form
        A, B, C = y2 - y1, x1 - x2, x2 * y1 - x1 * y2
        return A, B, C

    def move_line(self, distance: float | int):
        """
        Move the line in the direction perpendicular to it by a specified distance.

        :param distance: float | int. Distance by which to move the line.
        """
        # Delta C is derived by translating the line equation Ax + By + C = 0
        delta_C = -distance * self._norm
        self.C += delta_C

    def move_line_to_intersect_point(self, x: float | int, y: float | int) -> tuple[float, float]:
        """
        Move the line to intersect a given point (x, y).

        :param x: float | int. The x-coordinate of the point.
        :param y: float | int. The y-coordinate of the point.
        :return: tuple[float, float]. New A, B, C coefficients of the line.
        """
        # Calculate the new C such that the line goes through the point (x, y)
        self.C = -(self.A * x + self.B * y)
        return self.A, self.B, self.C

    def distance_from_point(self, x: float | int, y: float | int) -> float:
        """
        Calculate the distance of a point (x, y) from the line.

        :param x: float | int. The x-coordinate of the point.
        :param y: float | int. The y-coordinate of the point.
        :return: float. Distance of the point from the line.
        """
        return abs(self.point_line_position(x=x, y=y)) / self._norm

    def distances_from_points(self, points: np.ndarray) -> np.ndarray:
        """
        Calculate the distances from each point in a numpy array to the line.

        :param points: np.ndarray. An array of shape (N, 2) containing N points.
        :return: np.ndarray. An array of shape (N,) containing the distances of each point to the line.
        """
        assert points.shape[1] == 2, "Input array must be of shape (N, 2)."

        # Calculate Ax + By + C for each point
        point_line_positions = self.A * points[:, 0] + self.B * points[:, 1] + self.C

        # Calculate the distance using the formula
        distances = np.abs(point_line_positions) / self._norm

        return distances

    def point_line_position(self, x: float | int, y: float | int) -> float:
        """
        Calculate the position of a point (x, y) relative to the line.
        The value will be positive, negative, or zero depending on the point's position.

        :param x: float | int. The x-coordinate of the point.
        :param y: float | int. The y-coordinate of the point.
        :return: float. Position value.
        """
        return self.A * x + self.B * y + self.C

    def get_intersection(self, other_line: _Line) -> tuple[float, float] | None:
        """
        Find the intersection point between this line and another line.

        :param other_line: _Line. The other line represented as an instance of the _Line class.
        :return: tuple[float, float] | None. Tuple representing the intersection point, or None if lines are parallel.
        """
        # Using Cramer's method to solve the system of equations formed by the two lines
        # A1*x + B1*y + C1 = 0 and A2*x + B2*y + C2 = 0
        det = self.A * other_line.B - other_line.A * self.B

        if det == 0:
            # Lines are parallel, no intersection
            return None

        x = (-self.C * other_line.B + other_line.C * self.B) / det
        y = (-self.A * other_line.C + other_line.A * self.C) / det

        return x, y


class QuadrilateralFitter:
    def __init__(self, polygon: np.ndarray | tuple | list | Polygon):
        """
        Constructor for initializing the QuadrilateralFitter object.

        :param polygon: np.ndarray. A NumPy array of shape (N, 2) representing the input polygon,
                              where N is the number of vertices.
        """
        
        if isinstance(polygon, Polygon):
            _polygon = polygon
            self._polygon_coords = np.array(polygon.exterior.coords, dtype=np.float32)
        else:
            if type(polygon) == np.ndarray:
                assert polygon.shape[1] == len(
                    polygon.shape) == 2, f"Input polygon must have shape (N, 2). Got {polygon.shape}"
                _polygon = Polygon(polygon)
                self._polygon_coords = polygon

            elif isinstance(polygon, (list, tuple)):
                # Checking if the list or tuple has sub-lists/tuples of length 2 (i.e., coordinates)
                assert all(isinstance(coord, (list, tuple)) and len(coord) == 2 for coord in
                           polygon), "Expected sub-lists or sub-tuples of length 2 for coordinates"
                _polygon = Polygon(polygon)
                self._polygon_coords = np.array(polygon, dtype=np.float32)
            else:
                raise TypeError(f"Unexpected input type: {type(polygon)}. Accepted are np.ndarray, tuple, "
                                f"list and shapely.Polygon")

        self.convex_hull_polygon = _polygon.convex_hull

        self._initial_guess = None

        self._line_equations = None
        self.fitted_quadrilateral = None

        self._expanded_line_equations = None
        self.expanded_fitted_quadrilateral = None

    def fit(self, simplify_polygons_larger_than: int|None = 10, start_simplification_epsilon: float = 0.1,
            max_simplification_epsilon: float = 0.5, simplification_epsilon_increment: float = 0.02) -> \
            tuple[tuple[float, float], tuple[float, float], tuple[float, float], tuple[float, float]]:
        """
        Fits an irregular quadrilateral around the input polygon. The quadrilateral is optimized to minimize
        the Intersection over Union (IoU) with the input polygon.

        This method performs the following steps:
        1. Computes the convex hull of the input polygon.
        2. Finds an initial quadrilateral that closely approximates the convex hull.
        3. Refines this initial quadrilateral to ensure it fully circumscribes the convex hull.

        Note: The input polygon should be of shape (N, 2), where N is the number of vertices.

        :param simplify_polygons_larger_than: int | None. If a number is specified, the method will make a
                        preliminar Douglas-Peucker simplification of the Convex Hull if it has more than
                        simplify_polygons_larger_than vertices. This will speed up the process, but may
                        lead to a sub-optimal quadrilateral approximation.
        :param start_simplification_epsilon: float. The initial simplification epsilon to use if
                        simplify_polygons_larger_than is not None (for Douglas-Peucker simplification).
        :param max_simplification_epsilon: float. The maximum simplification epsilon to use if
                        simplify_polygons_larger_than is not None (for Douglas-Peucker simplification).
        :param simplification_epsilon_increment: float. The increment in the simplification epsilon to use if
                        simplify_polygons_larger_than is not None (for Douglas-Peucker simplification).

        :return: A tuple containing four tuples, each of which has two float elements representing the (x, y)
                coordinates of the quadrilateral's vertices. The vertices are order clockwise.

        :raises AssertionError: If the input polygon does not have a shape of (N, 2).
        """
        self._initial_guess = self.__find_initial_quadrilateral(max_sides_to_simplify=simplify_polygons_larger_than,
                                                            start_simplification_epsilon=start_simplification_epsilon,
                                                            max_simplification_epsilon=max_simplification_epsilon,
                                                            simplification_epsilon_increment=simplification_epsilon_increment)
        self.fitted_quadrilateral = self.__finetune_guess()
        self.expanded_quadrilateral = self.__expand_quadrilateral()
        return self.fitted_quadrilateral


    def __find_initial_quadrilateral(self, max_sides_to_simplify: int | None = 10,
                                     start_simplification_epsilon: float = 0.1,
                                     max_simplification_epsilon: float = 0.5,
                                     simplification_epsilon_increment: float = 0.02,
                                     max_combinations: int = 300) -> Polygon:
        """
        Internal method to find the initial approximating quadrilateral based on the vertices of the Convex Hull.
        To find the initial quadrilateral, we iterate through all 4-vertex combinations of the Convex Hull vertices
        and find the one with the highest Intersection over Union (IoU) with the Convex Hull. It will ensure that
        it is the best possible quadrilateral approximation to the input polygon.
        :param max_sides_to_simplify: int|None. If a number is specified, the method will make a
                        preliminar Douglas-Peucker simplification of the Convex Hull if it has more than
                        max_sides_to_simplify vertices. This will speed up the process, but may
                        lead to a sub-optimal quadrilateral approximation.
        :param start_simplification_epsilon: float. The initial simplification epsilon to use if
                        max_sides_to_simplify is not None (for Douglas-Peucker simplification).
        :param max_simplification_epsilon: float. The maximum simplification epsilon to use if
                        max_sides_to_simplify is not None (for Douglas-Peucker simplification).
        :param simplification_epsilon_increment: float. The increment in the simplification epsilon to use if
                        max_sides_to_simplify is not None (for Douglas-Peucker simplification).

        :param max_combinations: int. The maximum number of combinations to try. If the number of combinations
                        is larger than this number, the method will only run random max_combinations combinations.

        :return: Polygon. A Shapely Polygon object representing the initial quadrilateral approximation.
        """
        best_iou, best_quadrilateral = 0., None  # Variable to store the vertices of the best quadrilateral
        convex_hull_area = self.convex_hull_polygon.area

        # Simplify the Convex Hull if it has more than simplify_polygons_larger_than vertices
        simplified_polygon = self.__simplify_polygon(polygon=self.convex_hull_polygon,
                                                     max_sides=max_sides_to_simplify,
                                                     initial_epsilon=start_simplification_epsilon,
                                                     max_epsilon=max_simplification_epsilon,
                                                     epsilon_increment=simplification_epsilon_increment)

        all_combinations = tuple(combinations(mapping(simplified_polygon)['coordinates'][0], 4))

        # Limit the number of combinations to max_combinations if it's too large, to speed up the process
        if len(all_combinations) > max_combinations:
            all_combinations = sample(all_combinations, max_combinations)

        # Iterate through 4-vertex combinations to form potential quadrilaterals
        for vertices_combination in all_combinations:
            current_quadrilateral = Polygon(vertices_combination)
            assert current_quadrilateral.is_valid, f"Quadrilaterals generated from an ordered Convex Hull should be " \
                                                       f"always valid."

            # Calculate the Intersection over Union (IoU) between the Convex Hull and the current quadrilateral
            iou = self.__iou(polygon1=self.convex_hull_polygon, polygon2=current_quadrilateral,
                             precomputed_polygon_1_area=convex_hull_area)

            if iou > best_iou:
                best_iou, best_quadrilateral = iou, current_quadrilateral
                if iou >= 1.:
                    assert iou == 1., f"IoU should never be > 1.0. Got{iou}"
                    break  # We found the best possible quadrilateral, so we can stop iterating

        assert best_quadrilateral is not None, "No quadrilateral was found. This should never happen."

        return best_quadrilateral

    def __finetune_guess(self) -> Polygon:
        """
        Internal method to finetune the initial quadrilateral approximation to adjust to the input polygon.
        The method works by deciding which point of the initial polygon belongs to which side of the input polygon
        and fitting a line to each side of the input polygon. The intersection points between the lines will
        be the vertices of the new quadrilateral.

        :return: Polygon. A Shapely Polygon object representing the finetuned quadrilateral.
        """

        initial_line_equations = self.__polygon_vertices_to_line_equations(polygon=self._initial_guess)
        # Calculate the distance between each vertex of the input polygon and each line of the quadrilateral
        distances = np.array(
            [line.distances_from_points(points=self._polygon_coords) for line in initial_line_equations],
            dtype=np.float32)
        # For each point, get the index of the closest line
        points_line_idx = np.argmin(distances, axis=0)

        self._line_equations = tuple(
            self.__linear_regression(self._polygon_coords[points_line_idx == i], initial_guess=initial_guess)
                for i, initial_guess in enumerate(initial_line_equations)
        )

        new_quadrilateral_vertices = self.__find_polygon_vertices_from_lines(line_equations=self._line_equations)
        return new_quadrilateral_vertices

    def __linear_regression(self, points: np.ndarray, initial_guess: _Line = None) -> _Line:
        """
        Internal method that fits a line from a set of points using linear regression.
        :param points: np.ndarray. A numpy array of shape (N, 2) representing the points to fit the line to. Format X,Y
        :param initial_guess: _Line. An initial guess for the line equation. If None, the method will use the
                        linear regression method to find the best possible line.
        :return: _Line. A _Line object representing the fitted line.
        """

        def perpendicular_distance(params, points: np.ndarray):
            a, b, c = params
            x, y = points[:, 0], points[:, 1]
            return np.sum(np.abs(a * x + b * y + c)) / np.sqrt(a * a + b * b)

        if initial_guess is None:
            initial_guess = (1., -1., 0.)
        else:
            initial_guess = (initial_guess.A, initial_guess.B, initial_guess.C)

        result = minimize(perpendicular_distance, initial_guess, args=(points,), method='Nelder-Mead')
        A, B, C = result.x

        return _Line(A=A, B=B, C=C)

    def __expand_quadrilateral(self) -> Polygon:
        """
        Internal method that expands the initial quadrilateral approximation to make sure it contains all the vertices
        of the input polygon Convex Hull.
        Method:
            1. Move each line in their orthogonal direction (outwards) until it contains (or intersects)
               all the points of the Convex Hull in its inward direction
            2. Find the intersection points between the lines to calculate the vertices of the
               new expanded quadrilateral

        :param quadrilateral: Polygon. A Shapely Polygon object representing the initial quadrilateral approximation.

        :return: Polygon. A Shapely Polygon object representing the expanded quadrilateral.
        """
        # 1. Move each line in their orthogonal direction (outwards) until it contains (or intersects)
        #    all the points of the Convex Hull in its inward direction
        line_equations = tuple(line.copy() for line in self._line_equations)
        for line in line_equations:
            self.__move_line_to_contain_all_points(line=line, polygon=self.convex_hull_polygon)
        # 3. Find the intersection points between the lines to calculate the vertices of the
        #    new expanded quadrilateral
        new_quadrilateral_vertices = self.__find_polygon_vertices_from_lines(line_equations=line_equations)
        return new_quadrilateral_vertices

    def __find_polygon_vertices_from_lines(self, line_equations: tuple[_Line]) -> tuple[tuple[float, float], ...]:
        """
        Internal method to calculate the vertices of a polygon from a tuple of line equations.

        :param line_equations: tuple[_Line]. A tuple of _Line objects representing the sides of the polygon.
        :return: tuple[tuple[float, float], ...]. A tuple of tuples representing the vertices of the polygon.
        """
        # Find the intersection between each line and its next one
        points = tuple(line1.get_intersection(other_line=line_equations[(i + 1) % len(line_equations)])
                     for i, line1 in enumerate(line_equations))
        # Order points clockwise
        points = self.__order_points_clockwise(pts=points)
        return points

    def __order_points_clockwise(self, pts: np.ndarray | tuple[tuple[float, float], ...]) -> tuple[tuple[float, float], ...]:

        as_np = isinstance(pts, np.ndarray)
        if not as_np:
            pts = np.array(pts, dtype=np.float32)
        # Calculate the center of the points
        center = np.mean(pts, axis=0)

        # Compute the angles from the center
        angles = np.arctan2(pts[:, 1] - center[1], pts[:, 0] - center[0])

        # Sort the points by the angles in ascending order
        sorted_pts = pts[np.argsort(angles)]

        if not as_np:
            sorted_pts = tuple(tuple(pt) for pt in sorted_pts)
        return sorted_pts

    @staticmethod
    def __polygon_vertices_to_line_equations(polygon: Polygon) -> tuple[_Line]:
        """
        Internal static method to convert Polygons to a tuple of line equations.

        :param polygon: Polygon. A Shapely Polygon object.
        :return: tuple[_Line]. A tuple of _Line objects representing the sides of the polygon.
        """
        assert isinstance(polygon, Polygon), f"Expected a Shapely Polygon, got {type(polygon)} instead."
        coords = polygon.exterior.coords
        # Loop through each pair of vertices to calculate the line equations (Last coord is same as first (Shapely))
        return tuple(_Line(x1=x1, y1=y1, x2=x2, y2=y2)
                     for (x1, y1), (x2, y2) in zip(coords[:-1], coords[1:]))

    def __move_line_to_contain_all_points(self, line: _Line, polygon: Polygon) -> bool:
        """
        Internal method to move a line until it contains all points in the Convex Hull.

        :param line: _Line. The line to be moved.
        :param polygon: Polygon. The polygon to be contained by the line after moving it.

        :return: bool. True if the line was moved, False otherwise.
        """
        centroid = polygon.centroid
        centroid_sign = self.__sign(x=line.point_line_position(x=centroid.x, y=centroid.y))
        assert centroid_sign != 0, "The centroid of the polygon should never be on the line."

        max_distance, best_point = 0., None

        for (x, y) in self.convex_hull_polygon.exterior.coords[:-1]:
            point_position = line.point_line_position(x=x, y=y)
            if self.__sign(x=point_position) != centroid_sign:
                distance = line.distance_from_point(x=x, y=y)
                if distance > max_distance:
                    max_distance, best_point = distance, (x, y)

        if best_point is not None:
            x, y = best_point
            line.move_line_to_intersect_point(x=x, y=y)
            return True
        return False


    # -------------------------------- HELPER METHODS -------------------------------- #

    def __simplify_polygon(self, polygon: Polygon, max_sides: int|None,
                           initial_epsilon: float = 0.1, max_epsilon: float = 0.5,
                           epsilon_increment: float = 0.02, iou_threshold = 0.8) -> Polygon:
        """
        Internal method to simplify a polygon using the Douglas-Peucker algorithm.
        :param polygon: Polygon. The polygon to simplify.
        :param max_sides: int|None. The maximum number of sides the polygon can have after simplification.
                            If None, no simplification will be performed.
        :param max_epsilon: float. The maximum tolerance value for the Douglas-Peucker algorithm.
        :param initial_epsilon: float. The initial tolerance value for the Douglas-Peucker algorithm.
        :param epsilon_increment: float. The incremental step for the tolerance value.

        :return: Polygon. The simplified polygon.
        """
        if max_sides is None or len(polygon.exterior.coords) - 1 <= max_sides:
            return polygon  # No simplification needed

        assert max_epsilon > 0., f"max_epsilon should be a float greater than 0. Got {max_epsilon}."
        assert initial_epsilon > 0., f"initial_epsilon should be a float greater than 0. Got {initial_epsilon}."
        assert epsilon_increment > 0., f"epsilon_increment should be a float greater than 0. Got {epsilon_increment}."

        simplified_polygon = polygon
        original_polygon_area = polygon.area

        epsilon = initial_epsilon
        while epsilon <= max_epsilon:
            # Simplify the polygon
            simplified_polygon_unconfirmed = simplified_polygon.simplify(epsilon, preserve_topology=True)
            n_sides = len(simplified_polygon_unconfirmed.exterior.coords) - 1
            # If the polygon has less than 4 sides, it becomes invalid, get the previous one
            if n_sides < 4:
                break
            # If the polygon has less than max_sides, we have it. Return it or the previous one depending on the IoU
            elif len(simplified_polygon_unconfirmed.exterior.coords) - 1 <= max_sides:
                iou = self.__iou(polygon1=simplified_polygon_unconfirmed, polygon2=self.convex_hull_polygon,
                                 precomputed_polygon_1_area=original_polygon_area)
                # If the IoU is beyond the threshold, we accept the polygon. Otherwise, return the previous one
                if iou > iou_threshold:
                    # We accept the polygon
                    simplified_polygon = simplified_polygon_unconfirmed
                return simplified_polygon
            else:
                # If the polygon has more than max_sides, that's our best guess so far, but keep trying
                simplified_polygon = simplified_polygon_unconfirmed
                epsilon += epsilon_increment

        return simplified_polygon

    def __iou(self, polygon1: Polygon, polygon2: Polygon, precomputed_polygon_1_area: float | None = None) -> float:
        """
        Calculate the Intersection over Union (IoU) between two polygons.

        :param polygon1: Polygon. The first polygon.
        :param polygon2: Polygon. The second polygon.
        :param precomputed_polygon_1_area: float|None. The area of the first polygon. If None, it will be computed.
        :return: float. The IoU value.
        """
        if precomputed_polygon_1_area is None:
            precomputed_polygon_1_area = polygon1.area
        # Calculate the intersection and union areas
        intersection = polygon1.intersection(polygon2).area
        union = precomputed_polygon_1_area + polygon2.area - intersection
        # Return the IoU value
        return (intersection / union) if union != 0. else 0.

    @staticmethod
    def __sign(x: int | float) -> int:
        """
        Return the sign of a number.
        :param x: float. The number to check.
        :return: int. 1 if x > 0, -1 if x < 0, 0 if x == 0.
        """
        return 1 if x > 0 else (-1 if x < 0 else 0)
    

class Point:

    def __init__(self, x: float, y: float):
        self.x = x
        self.y = y


    def __repr__(self):
        return f'[{self.x}, {self.y}]'


    def to_np(self):
        return np.asarray([self.x, self.y])
    
    def to_list(self):
        return [self.x, self.y]

    def segment(self, direction):
        '''
        Return directed segment, which has the direction of @direction, formed by starting at @self 
        that has length of 1

        @direction: float, in [0, 2 * pi)

        Compute the coordinate of the end point b by forming right triangle abc with right angle 
        at vertex c
        '''
        b = None
        acute_angle = None
        signs = None

        if direction <= np.pi / 2:
            signs = [1, 1]
            acute_angle = direction
        elif direction <= np.pi:
            signs = [-1, 1]
            acute_angle = np.pi - direction
        elif direction <= 3 * np.pi / 2:
            signs = [-1, -1]
            acute_angle = direction - np.pi
        else:
            signs = [1, -1]
            acute_angle = 2 * np.pi - direction

        bc_length = np.sin(acute_angle)
        ac_length = np.cos(acute_angle)
        b = Point(self.x + signs[0] * ac_length, self.y + signs[1] * bc_length)

        return Segment(self, b)



class Segment:
    '''directed straight line'''

    def __init__(self, start_point, end_point):
        '''
        @start_point, @end_point: Point
        '''
        self.start_point = start_point
        self.end_point = end_point
        self.line = self.line_formula()


    def direction(self):
        '''
        Return angle between right oriented horizontal ray (0, 1) and @self, in [0, 2 * pi)
        Using a.b = |a||b|cos(theta)
        '''
        s = np.array([self.end_point.x - self.start_point.x, self.end_point.y - self.start_point.y])

        theta = np.arctan2(s[1], s[0])
        
        return (theta / np.pi % 2) * np.pi


    def angle(self, t):
        '''
        Return the angle between @self and @t

        @t: Segment
        '''
        return t.direction() - self.direction()


    def perpendicular(self, p):
        '''
        Return the segment pq perpendicular to @self with source @p and target on line(@self)

        @p: Point

        line(@self = [a, b]): (a.y - b.y)x - (a.x - b.x)y = a.y * b.x - a.x * b.y
        q in line(@self) -> (a.y - b.y)q.x - (a.x - b.x)q.y = a.y * b.x - a.x * b.y
        pq perpendicular to @self -> [p.x - q.x, p.y - q.y].[a.x - b.x, a.y - b.y] = 0
                                or (a.x - b.x)q.x + (a.y - b.y)q.y = (a.x - b.x)p.x + (a.y - b.y)p.y
        '''
        a = self.start_point
        b = self.end_point

        coefficient_matrix = [[self.line[0], self.line[1]], [b.x - a.x, b.y - a.y]]
        ordinate = [-self.line[2], (b.x - a.x) * p.x + (b.y - a.y) * p.y]
        q = np.linalg.solve(coefficient_matrix, ordinate)
        q = Point(q[0], q[1])

        return Segment(p, q)


    def length(self):
        '''
        Return the length of @self
        '''
        return np.sqrt((self.start_point.x - self.end_point.x) ** 2 + 
            (self.start_point.y - self.end_point.y) ** 2)


    def line_formula(self):
        '''
        Return formula of the line passing @self

        line: mx + ny + p = 0
        '''
        a = self.start_point
        b = self.end_point

        m = a.y - b.y
        n = b.x - a.x
        p = a.x * b.y - a.y * b.x 

        return (m, n, p)


    def intersection(self, g):
        '''
        Return the intersector of @self and @g

        @g: Segment
        '''

        intersect = not (abs(self.direction() - g.direction()) < 1e-7 
            or abs(abs(self.direction() - g.direction()) - np.pi) < 1e-7)
        intersector = None

        if intersect:
            coefficient_matrix = [[self.line[0], self.line[1]], [g.line[0], g.line[1]]]
            ordinate = [-self.line[2], -g.line[2]]

            intersector = np.linalg.solve(coefficient_matrix, ordinate)
            intersector = Point(intersector[0], intersector[1])

        return intersector


    def __repr__(self):
        return f'[{self.start_point}, {self.end_point}]'


class Evtype:

    def __init__(self, v, e):
        '''
        @v: Point
        @e: Segement
        '''

        self.v = v # Point
        self.e = e # Segment


    def edge(self):
        return self.e


    def vertex(self):
        return self.v


    def angle(self, p):
        '''
        Return the angle by the edges of 2 antipodal edge-vertex pairs

        @p: Evtype
        '''
        return self.e.angle(p.e)


    def width(self):
        '''
        Return the width of an antipodal pair (e, v), which is the distance of v
        to its orthogonal projection on the supporting line of e
        '''
        return self.e.perpendicular(self.v).length()


    def __repr__(self):
        return f'{self.e}, {self.v}'


class Parallelogram:

    def __init__(self, z1, z2):
        '''
        @z1, @z2: Evtype
        '''
        self.a = None   # Point
        self.b = None   # Point
        self.c = None   # Point
        z1_direction = z1.e.direction()
        z2_direction = z2.e.direction()
        self.a = z1.e.intersection(z2.e)
        self.b = z1.e.intersection(z2.v.segment(z2_direction))
        self.c = z2.v.segment(z2_direction).intersection(z1.v.segment(z1_direction))


    def drawable(self):
        return not (self.a == None or self.b == None or self.c == None)


    def d(self):
        '''
        Return fourth point from 3 other pts of the parallelogram
        The order in clockwise: a - b - c - d
        -> od = oa - bc
        '''
        return Point(self.a.x - self.b.x + self.c.x, self.a.y - self.b.y + self.c.y)


    def angle(self):
        return Segment(self.a, self.b).angle(Segment(self.b, self.c))


    def area(self):
        theta = abs(self.angle()) % np.pi
        return Segment(self.a, self.b).length() * Segment(self.b, self.c).length() \
                * np.sin(theta)


    def __repr__(self):
        return f'{self.a}, {self.b}, {self.c}, {self.d()}'


def antipodal_pairs(vertices):
    '''
    Traverse through every pair of adjacent vertices, find the point which combines to
    the current segment to create an antipodal pair.

    @vertices: list[Point]
        list of vertices of the polygon
    '''
    antipodal_evs = []

    for i in range(len(vertices)):
        v1 = vertices[i]
        v2 = vertices[(i+1) % len(vertices)]

        max_distance = 0
        antipodal_ev = None

        for v in vertices:
            if not ((v.x == v1.x and v.y == v1.y) or (v.x == v2.x and v.y == v2.y)):
                ev = Evtype(v, Segment(v1, v2))
                distance = ev.width()
                if distance > max_distance:
                    max_distance = distance
                    antipodal_ev = ev

        antipodal_evs.append(antipodal_ev)

    return antipodal_evs


def simple_mep(evs):
    '''
    @evs: list[Evtype]
        list of antipodal pairs of the polygon
    '''
    min_area = 1e10
    mep = None
    ev1 = None
    ev2 = None

    for i in range(len(evs) - 1):
        for j in range(i, len(evs)):
            pargram = Parallelogram(evs[i], evs[j])

            if not pargram.drawable():
                continue

            area = pargram.area()
            if area < min_area:
                min_area = area
                mep = pargram
                ev1 = evs[i]
                ev2 = evs[j]

    return mep, ev1, ev2

class ParallelogramFitter:
    def __init__(self, polygon: np.ndarray | tuple | list, convex_hull = True ):

        if type(polygon) == np.ndarray:
            assert polygon.shape[1] == len(polygon.shape) == 2, f"Input polygon must have shape (N, 2). Got {polygon.shape}"
            self._polygon_coords = polygon

        elif isinstance(polygon, (list, tuple)):
            # Checking if the list or tuple has sub-lists/tuples of length 2 (i.e., coordinates)
            assert all(isinstance(coord, (list, tuple)) and len(coord) == 2 for coord in
                        polygon), "Expected sub-lists or sub-tuples of length 2 for coordinates"
            self._polygon_coords = np.array(polygon, dtype=np.float32)
        else:
            raise TypeError(f"Unexpected input type: {type(polygon)}. Accepted are np.ndarray, tuple, "
                            f"list and shapely.Polygon")

        if convex_hull:
            self.convex_hull_polygon = cv2.convexHull(self._polygon_coords)
        else:
            self.convex_hull_polygon = self._polygon_coords # _polygon is already a convex hull (user knows)

    def fit(self, ):

        if self.convex_hull_polygon.shape[0] < 3:
            return [np.NaN, np.NaN, np.NaN, np.NaN]
        cvx_polygon = [Point(pt[0, 0], pt[0, 1]) for pt in self.convex_hull_polygon]
        antipodal_evs = antipodal_pairs(cvx_polygon)
        min_ep, ev1, ev2 = simple_mep(antipodal_evs)
        a, b, c, d = min_ep.a.to_list(), min_ep.b.to_list(), min_ep.c.to_list(), min_ep.d().to_list()
        return [a, b, c, d]