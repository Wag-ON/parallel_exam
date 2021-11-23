import cv2
import numpy as np
import math
import copy
from rdp import rdp


if __name__ == "__main__":
    from polygonization_method import PolygonizationMethod
    import stuff
else:
    from utils.methods.polygonization_method import PolygonizationMethod
    from utils.methods import stuff


class GridMethod(PolygonizationMethod):
    def __init__(self):
        super().__init__()

    def get_best_alternate_node(self, given_point, grid_step, grid):
        best_dist = 1e9
        best_node = given_point
        for y in range(given_point[0] - grid_step, given_point[0] + grid_step):
            for x in range(given_point[1] - grid_step, given_point[1] + grid_step):
                if grid[x][y] == 255:
                    dist_to_alternate_node = stuff.calc_dist([y, x], given_point)
                    if dist_to_alternate_node < best_dist:
                        best_dist = dist_to_alternate_node
                        best_node = [y, x]

        return best_node

    def create_grid(self, grid_step=10, angle=0, shuffle_x=0, shuffle_y=0, shape=(0, 0)):
        """
        Create grid with given parameters
        grid_step = distance between nodes of grid in pixels
        angle = angle between the Oy-axis and grid (counterclockwise)
        shuffle_x, shuffle_y = offsets from start of coordinates
        shape = house.shape

        Return ndarray H * W
        where pixels with 255 value are nodes, rest are 0
        """

        grid = np.zeros(shape, np.uint8)
        height, width = shape

        angle = -angle

        for y in range(0 + shuffle_y, max(2 * height, 2 * width), grid_step):
            for x in range(-height + shuffle_x, width, grid_step):

                new_x = int(x * math.cos(math.radians(angle)) - y * math.sin(math.radians(angle)) + 0.5)
                new_y = int(y * math.cos(math.radians(angle)) + x * math.sin(math.radians(angle)) + 0.5)

                if 0 <= new_y < height and 0 <= new_x < width:
                    grid[new_y][new_x] = 255

        # plt.imshow(grid)
        # plt.show()
        return grid

    def node_is_exist_in_polygonized(self, new_point, points):
        """
        Checks if any points with coordinates of new_point is exist in polygon

        points is polygon in following format:
        polygon = [point_1, ... point_m]
        where point = [x, y]

        new_point in format [x, y]
        """

        for i in range(0, len(points)):
            if new_point[0] == points[i][0][0] and new_point[1] == points[i][0][1]:
                return True
        return False

    def try_grid(self, grid_step, angle, shuffle_x, shuffle_y, shape, house, house_contour):
        """
        Trying one variant of grid with given parameters
        grid_step = distance between nodes of grid in pixels
        angle = angle between the Oy-axis and grid
        shuffle_x, shuffle_y = offsets from start of coordinates
        shape = hose.shape
        house = binary mask in ndarray H * W
        contour- 3 dimensional ndarray [point_amount x 1 x 2]
        where contour[point_t][0][0] - x coordinate, contour[point_t][0][1] - y coordinate

        Returns dictionary, where
        ['len'] = number of points in the best contour
        ['iou'] = IOU score between best contour and house mask
        ['grid_step'] =  grid_step in pixels
        ['polygonized_points'] = polygon = [point_1, ... point_m] where point = [x, y]
        ['house_pic'] = binary mask in ndarray H * W
        """

        grid = self.create_grid(
            grid_step=grid_step,
            angle=angle,
            shuffle_x=shuffle_x,
            shuffle_y=shuffle_y,
            shape=shape)

        polygonized_points = []
        house_polygonized = np.zeros(house.shape, np.uint8)
        for current_node in house_contour:
            best_node = self.get_best_alternate_node(current_node[0], grid_step, grid)  # [y, x]
            if not self.node_is_exist_in_polygonized(best_node, polygonized_points):
                polygonized_points.append([best_node])

        cv2.drawContours(house_polygonized, [np.array(polygonized_points)], contourIdx=-1, color=255, thickness=-1)
        iou = stuff.calc_iou(house, house_polygonized)

        return {'polygonized_points': polygonized_points,
                'iou': iou,
                'len': len(polygonized_points),
                'grid_step': grid_step,
                'house_pic': house_polygonized
                }

    def grids_bruteforce(self, house, house_contour, angle):
        """
        Choosing best polygonized contour with bruteforce

        house = binary mask in ndarray H * W
        contour- 3 dimensional ndarray [point_amount x 1 x 2]
        where contour[point_t][0][0] - x coordinate, contour[point_t][0][1] - y coordinate

        angle = angle between the Oy-axis and longest side in Peuckered version of house contour

        Returns polygon in following format:
        polygon = [point_1, ... point_m]
        where point = [x, y]

        """

        best_iou = 0

        variants = []
        grid_params = []

        for grid_step in range(11, 16, 1):
            for shuffle_x in [0, grid_step // 2]:
                for shuffle_y in [0, grid_step // 2]:
                    grid_params.append([grid_step, shuffle_x, shuffle_y])

        for grid_param in grid_params:
            variants.append(self.try_grid(
                grid_step=grid_param[0],
                angle=angle,
                shuffle_x=grid_param[1],
                shuffle_y=grid_param[2],
                shape=house.shape,
                house=house,
                house_contour=house_contour))

        variants = sorted(variants, key=lambda k: k['len'])
        best_variant = variants[0]
        for i in range(1, len(variants) // 2):
            if variants[i]['len'] <= 1.3 * variants[0]['len']:
                if variants[i]['iou'] > best_iou:
                    best_iou = variants[i]['iou']
                    best_variant = variants[i]

        return self.delete_triangles(best_variant['polygonized_points'])

    def delete_triangles(self, polygon):
        """
        Deleting "triangle noise" from polygon

        Returns polygon in following format:
        polygon = [point_1, ... point_m]
        where point = [x, y]

        """

        this_polygon = copy.deepcopy(polygon)

        i = 1
        while i < len(this_polygon) - 1:
            a = this_polygon[i - 1][0]
            b = this_polygon[i][0]
            c = this_polygon[i + 1][0]
            ab = stuff.calc_squared_dist(a, b)
            ac = stuff.calc_squared_dist(a, c)
            bc = stuff.calc_squared_dist(b, c)
            # print(ab, ac, bc)
            if (ab < bc and ac < bc) or (ac < ab and bc < ab):
                this_polygon.pop(i)
            i += 1

        a = this_polygon[len(this_polygon) - 2][0]
        b = this_polygon[len(this_polygon) - 1][0]
        c = this_polygon[0][0]
        ab = stuff.calc_squared_dist(a, b)
        ac = stuff.calc_squared_dist(a, c)
        bc = stuff.calc_squared_dist(b, c)

        if (ab < bc and ac < bc) or (ac < ab and bc < ab):
            this_polygon.pop(len(this_polygon) - 1)

        a = this_polygon[len(this_polygon) - 1][0]
        b = this_polygon[0][0]
        c = this_polygon[1][0]
        ab = stuff.calc_squared_dist(a, b)
        ac = stuff.calc_squared_dist(a, c)
        bc = stuff.calc_squared_dist(b, c)

        if (ab < bc and ac < bc) or (ac < ab and bc < ab):
            this_polygon.pop(0)

        return this_polygon

    def process_single_house(self, house, x_offset_in_original, y_offset_in_original):
        """
        Handle_single_house to find best polygonized contour

        house = binary mask in ndarray H * W
        x_offset_in_original,
        y_offset_in_original = coordinates of up-left point of house bounding box in original area mask

        Return contour in following format:
        where contour - 3 dimensional ndarray [point_amount x 1 x 2]
        where contour[point_t][0][0] - x coordinate, contour[point_t][0][1] - y coordinate

        """

        padding = 70
        house = np.pad(house, ((padding, padding), (padding, padding)), 'constant')

        contour, _ = cv2.findContours(house, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)[-2:]
        area = cv2.contourArea(contour[0])
        peucker_epsilon = max(area // 10000, 5)
        peuckered = rdp(contour[0], epsilon=peucker_epsilon)

        longest_point_1, longest_point_2, dist = stuff.longest_side(peuckered)
        angle = stuff.calc_angle_of_line(longest_point_1, longest_point_2)

        polygon_local = self.grids_bruteforce(house, contour[0], angle)

        polygon_global = []

        for point in polygon_local:
            polygon_global.append(
                [[point[0][0] + x_offset_in_original - padding, point[0][1] + y_offset_in_original - padding]])

        contour_global = np.array(polygon_global)

        return contour_global

    def handle_single_house(self, single_contour):
        house, x_top_left_bb, y_top_left_bb = super().prepare_single_house(single_contour)
        polygonized_contour = self.process_single_house(house, x_top_left_bb, y_top_left_bb)

        return polygonized_contour
