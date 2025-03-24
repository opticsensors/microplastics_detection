import numpy as np
import cv2
import sympy
import math
from fitters import ParallelogramFitter, QuadrilateralFitter


def simplify_contour(contour, n_corners=4):

    n_iter, max_iter = 0, 100
    lb, ub = 0., 1.

    while True:
        n_iter += 1
        if n_iter > max_iter:
            return contour

        k = (lb + ub)/2.
        eps = k*cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, eps, True)

        if len(approx) > n_corners:
            lb = (lb + ub)/2.
        elif len(approx) < n_corners:
            ub = (lb + ub)/2.
        else:
            return approx

def vertices_to_midpoints(vertices):
    v1 = vertices[0]
    v2 = vertices[1]
    v3 = vertices[2]
    v4 = vertices[3]
    midpoint1 = (v1 + v2) / 2  # Edge between points 1 and 2
    midpoint2 = (v2 + v3) / 2  # Edge between points 2 and 3
    midpoint3 = (v3 + v4) / 2  # Edge between points 3 and 4
    midpoint4 = (v4 + v1) / 2  # Edge between points 4 and 1
    return np.vstack([midpoint1,midpoint2,midpoint3,midpoint4])

def midpoints_to_axis(midpoints):
    return (midpoints[0], midpoints[3]), (midpoints[2], midpoints[4])

def quad_fit(binary_img, method, n=4,):

    contours, _ = cv2.findContours(binary_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnt = contours[0]


    if method == 'sym':
        hull = cv2.convexHull(cnt)
        hull = np.array(hull).reshape((len(hull), 2))

        # to sympy land
        hull = [sympy.Point(*pt) for pt in hull]

        # run until we cut down to n vertices
        while len(hull) > n:
            best_candidate = None

            # for all edges in hull ( <edge_idx_1>, <edge_idx_2> ) ->
            for edge_idx_1 in range(len(hull)):
                edge_idx_2 = (edge_idx_1 + 1) % len(hull)

                adj_idx_1 = (edge_idx_1 - 1) % len(hull)
                adj_idx_2 = (edge_idx_1 + 2) % len(hull)

                edge_pt_1 = sympy.Point(*hull[edge_idx_1])
                edge_pt_2 = sympy.Point(*hull[edge_idx_2])
                adj_pt_1 = sympy.Point(*hull[adj_idx_1])
                adj_pt_2 = sympy.Point(*hull[adj_idx_2])

                subpoly = sympy.Polygon(adj_pt_1, edge_pt_1, edge_pt_2, adj_pt_2)
                angle1 = subpoly.angles[edge_pt_1]
                angle2 = subpoly.angles[edge_pt_2]

                # we need to first make sure that the sum of the interior angles the edge
                # makes with the two adjacent edges is more than 180°
                if sympy.N(angle1 + angle2) <= sympy.pi:
                    continue

                # find the new vertex if we delete this edge
                adj_edge_1 = sympy.Line(adj_pt_1, edge_pt_1)
                adj_edge_2 = sympy.Line(edge_pt_2, adj_pt_2)
                intersect = adj_edge_1.intersection(adj_edge_2)[0]

                # the area of the triangle we'll be adding
                area = sympy.N(sympy.Triangle(edge_pt_1, intersect, edge_pt_2).area)
                # should be the lowest
                if best_candidate and best_candidate[1] < area:
                    continue

                # delete the edge and add the intersection of adjacent edges to the hull
                better_hull = list(hull)
                better_hull[edge_idx_1] = intersect
                del better_hull[edge_idx_2]
                best_candidate = (better_hull, area)

            if not best_candidate:
                raise ValueError("Could not find the best fit n-gon!")

            hull = best_candidate[0]

        # back to python land
        vertices = [(int(x), int(y)) for x, y in hull]

    elif method == 'quad':

        cnt=np.squeeze(cnt)
        vertices = QuadrilateralFitter(polygon=cnt).fit()
        vertices = np.array(vertices).astype(np.int32)


    elif method == 'para':

        cnt=np.squeeze(cnt) 
        vertices = ParallelogramFitter(polygon=cnt, convex_hull=True).fit()
        vertices = np.array(vertices).astype(np.int32)

    
    elif method == 'poly':
        vertices = simplify_contour(cnt)
        vertices = np.squeeze(vertices)

    
    elif method == 'rect':

        rect = cv2.minAreaRect(cnt)
        # Get the corner points of the rectangle
        vertices = cv2.boxPoints(rect)
        vertices = np.int64(vertices)  # Convert to integer points

    
    elif method == 'ellipse':
        ellipse = cv2.fitEllipse(cnt)
        (xc,yc),(d1,d2),angle = ellipse

        # circle at center
        xc, yc = ellipse[0]

        # major axis line 
        rmajor = max(d1,d2)/2
        if angle > 90:
            angle = angle - 90
        else:
            angle = angle + 90
        x1 = xc + math.cos(math.radians(angle))*rmajor
        y1 = yc + math.sin(math.radians(angle))*rmajor
        x3 = xc + math.cos(math.radians(angle+180))*rmajor
        y3 = yc + math.sin(math.radians(angle+180))*rmajor

        # minor axis line
        rminor = min(d1,d2)/2
        if angle > 90:
            angle = angle - 90
        else:
            angle = angle + 90
        x2 = xc + math.cos(math.radians(angle))*rminor
        y2 = yc + math.sin(math.radians(angle))*rminor
        x4 = xc + math.cos(math.radians(angle+180))*rminor
        y4 = yc + math.sin(math.radians(angle+180))*rminor

        v1 = (x1+x4-xc, y1+y4-yc)
        v2 = (x1+x2-xc, y1+y2-yc)
        v3 = (x2+x3-xc, y2+y3-yc)
        v4 = (x3+x4-xc, y3+y4-yc)
        vertices = np.array((v1,v2,v3,v4)).astype(np.int32)
    
    else:
        raise ValueError("method not supported!")
        
    vertices = np.array(vertices)
    return vertices_to_midpoints(vertices)

def max_min_fit(binary_img):

    contours, _ = cv2.findContours(binary_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    cnt = contours[0]
    cnt = np.squeeze(cnt)

    M = cv2.moments(cnt)
    cx = M["m10"] / M["m00"]
    cy = M["m01"] / M["m00"]
    
    centroid = np.array([cx,cy])

    distances = np.linalg.norm(centroid - cnt, axis=1)

    # Find the point furthest from the centroid (pmax)
    max_distance_index = np.argmax(distances)
    pmax = cnt[max_distance_index]
    
    # Find the point closest to the centroid (pmin)
    min_distance_index = np.argmin(distances)
    pmin = cnt[min_distance_index]
    
    # Find the point furthest from pmax
    distances2 = np.linalg.norm(pmax - cnt, axis=1)
    max_distance2_index = np.argmax(distances2)
    pmax2 = cnt[max_distance2_index]

    target_angle = np.pi
    vectors = cnt - centroid
    
    v_min = pmin - centroid
    
    # Calculate the angle between v_min and all other vectors
    dot_products = np.sum(v_min * vectors, axis=1)
    magnitudes = np.linalg.norm(v_min) * np.linalg.norm(vectors, axis=1)
    angles = np.arccos(np.clip(dot_products / magnitudes, -1.0, 1.0))
    
    # Find the point with the angle closest to 180 degrees (pi radians)
    target_angle = np.pi
    angle_differences = np.abs(angles - target_angle)
    pmin2_index = np.argmin(angle_differences)
    pmin2 = cnt[pmin2_index]

    return np.vstack([pmax,pmin,pmax2,pmin2])

def draw_midpoints_fit(img, midpoints, scale=1):
    midpoint1, midpoint2, midpoint3, midpoint4 = midpoints
    
    # Calculate lengths
    len1 = np.linalg.norm(midpoint1-midpoint3)*scale
    len2 = np.linalg.norm(midpoint2-midpoint4)*scale
    
    # Determine colors based on which axis is longer
    if len1 > len2:
        color1, color2 = (0, 0, 255), (255, 0, 0)  # len1 is red, len2 is blue
    else:
        color1, color2 = (255, 0, 0), (0, 0, 255)  # len1 is blue, len2 is red
    
    # Text positions
    p1 = midpoint1 if midpoint1[0] > midpoint3[0] else midpoint3
    p2 = midpoint2 if midpoint2[0] > midpoint4[0] else midpoint4
    
    img_to_save = img.copy()
    
    # Draw first axis
    cv2.line(img_to_save, (int(midpoint1[0]), int(midpoint1[1])), 
             (int(midpoint3[0]), int(midpoint3[1])), color1, thickness=3)
    cv2.putText(img_to_save, f"{len1:.2f} mm", (int(p1[0])+20, int(p1[1])-3),
               cv2.FONT_HERSHEY_SIMPLEX, 1, color1, 3)
    
    # Draw second axis
    cv2.line(img_to_save, (int(midpoint2[0]), int(midpoint2[1])), 
             (int(midpoint4[0]), int(midpoint4[1])), color2, thickness=3)
    cv2.putText(img_to_save, f"{len2:.2f} mm", (int(p2[0])+20, int(p2[1])-3),
               cv2.FONT_HERSHEY_SIMPLEX, 1, color2, 3)
    
    return img_to_save, len1, len2

def compute_size_given_axis_len(len_axis_y):
    if len_axis_y >= 999 and len_axis_y <= 4999:
        return "1000-4999"
    elif len_axis_y >= 5000 and len_axis_y <= 24999:
        return ">5 mm"
    elif len_axis_y >= 25000:
        return ">2.5 cm"
    elif len_axis_y < 300:
        return "<300"
    else:
        return "300-999"


