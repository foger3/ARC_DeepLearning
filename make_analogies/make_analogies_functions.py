import numpy as np
import matplotlib.pyplot as plt
from scipy import ndimage
import itertools


def paint_corner(image: np.array, furthest = True) -> np.array:
    height, width = image.shape
    all_coords = list(itertools.product(list(range(9)), repeat=2))
    if furthest:
        selected_distance = -np.Inf
    else:
        selected_distance = np.Inf
    furthest_corners = []
    corners = [(0,0),(height-1, 0),(0, width-1),( height-1, width-1)]
    for corner in corners:
        y = corner[0]
        x = corner[1]
        diss = []
        for coord_pair in all_coords:
            if image[coord_pair] == 255:
                vert_dist = np.max([y, coord_pair[0]]) - np.min([y, coord_pair[0]])
                hor_dist = np.max([x, coord_pair[1]]) - np.min([x, coord_pair[1]])
                diss.append(vert_dist + hor_dist)
        distance = np.min(diss)
        if distance == selected_distance:
            furthest_corners.append(corner)
        if distance > selected_distance and furthest:
            selected_distance = distance
            furthest_corners = [corner]
        if distance < selected_distance and not furthest:
            selected_distance = distance
            furthest_corners = [corner]
    new_image = np.copy(image)
    for corner in furthest_corners:
        new_image[corner[0], corner[1]] = 255
    return new_image


def paint_edge(image: np.array, coord: dict, furthest = True) -> np.array:
    height, width = image.shape
    top_distance = coord["top"]
    bottom_distance = height - coord["bottom"]
    left_distance = coord["left"]
    right_distance = width - coord["right"]
    if furthest:
        selected_distance = max(top_distance, bottom_distance, left_distance, right_distance)
    else:
        selected_distance = min(top_distance, bottom_distance, left_distance, right_distance)
    new_image = np.copy(image)
    if top_distance == selected_distance:
        new_image[0, :] = 255
    if bottom_distance == selected_distance:
        new_image[height-1, :] = 255
    if left_distance == selected_distance:
        new_image[:, 0] = 255
    if right_distance == selected_distance:
        new_image[:, width-1] = 255
    return new_image


def create_image_with_white_rectangle(coord: dict, img_size: int) -> np.array:
    black_image = np.zeros((img_size, img_size), dtype=np.uint8)
    black_image[coord["top"]:coord["bottom"], coord["left"]:coord["right"]] = 255
    return black_image


def invalid_matrix(matrix: np.array, height: int, width:int, number_rectangles: int) -> np.array:
    if matrix.shape != (height, width):
        return True #if wrong shape
    top_left_value = matrix[0, 0]
    if not np.all(matrix[0, :] == top_left_value) or not np.all(matrix[-1, :] == top_left_value) or not np.all(matrix[:, 0] == top_left_value) or not np.all(matrix[:, -1] == top_left_value):
        return True #if figure(s) touches edges
    labels, num_features = ndimage.label(matrix)
    if num_features != number_rectangles:
        return True #if wrong number of visible figures
    return False


def invert_colors(image: np.array) -> np.array:
    inverted_image = 255 - image
    return inverted_image


def plot_double_trio(double_trio: np.array) -> np.array:
    plt.figure(figsize=(12, 4))
    fig, axs = plt.subplots(2, 3)
    ind = 0
    for i in range(2):
        for j in range(3):
            axs[i, j].imshow(double_trio[ind], cmap='gray')  # Replace 'imshow' with your preferred plotting function
            if ind == 5:
                axs[i, j].set_title(f'TARGET')
            ind += 1


def mirror_image(image: np.array, horizontal=True) -> np.array:
    if horizontal:
        mirrored_image = np.fliplr(image)
    else:
        mirrored_image = np.flipud(image)
    return mirrored_image


def move_rectangle(image: np.array, horizontal_movement: int, vertical_movement: int) -> np.array:
    height, width = image.shape
    # Create a new black image with the same dimensions as the input image
    new_image = np.zeros((height, width), dtype=np.uint8)
    non_zero_coords = np.argwhere(image > 0)
    min_y, min_x = non_zero_coords.min(axis=0)
    max_y, max_x = non_zero_coords.max(axis=0)
    new_min_y = min_y + vertical_movement
    new_max_y = max_y + vertical_movement
    new_min_x = min_x + horizontal_movement
    new_max_x = max_x + horizontal_movement
    new_image[new_min_y:new_max_y+1, new_min_x:new_max_x+1] = 255
    return new_image


def resize_rectangle(image: np.array, coord: dict, top_add: int, bottom_add: int, left_add: int, right_add: int) -> np.array:
    new_coord ={}
    new_coord["top"] = np.max([0, coord["top"] - top_add])
    new_coord["left"]= np.max([0, coord["left"] - left_add])
    new_coord["right"] = np.min([image.shape[1], coord["right"]+ right_add])
    new_coord["bottom"] = np.min([image.shape[0], coord["bottom"] + bottom_add])
    new_image = create_image_with_white_rectangle(new_coord, img_size=image.shape[0])
    return new_image


def rotate_image(image: np.array, rotation_angle_degrees: int) -> np.array: #90, 180, 270
    if rotation_angle_degrees not in [90, 180, 270]:
        raise ValueError("Rotation angle must be 90, 180, or 270 degrees.")
    if rotation_angle_degrees == 90:
        rotated_image = np.rot90(image, k=1)
    elif rotation_angle_degrees == 180:
        rotated_image = np.rot90(image, k=2)
    else:
        rotated_image = np.rot90(image, k=3)
    return rotated_image


def draw_shadows(image: np.array, coord: dict, reverse: bool) -> np.array:
    new_image = np.zeros((image.shape[0], image.shape[1]), dtype=np.uint8)
    if reverse:
        new_image[0:coord["top"], image.shape[1]-1] = 255
        new_image[coord["bottom"]:image.shape[0], image.shape[1]-1] = 255
        new_image[0:coord["top"], 0] = 255
        new_image[coord["bottom"]:image.shape[0], 0] = 255
        new_image[0, 0:coord["left"]] = 255
        new_image[0, coord["right"]:image.shape[1]] = 255
        new_image[image.shape[0]-1, 0:coord["left"]] = 255
        new_image[image.shape[0]-1, coord["right"]:image.shape[1]] = 255
    else:
        new_image[coord["top"]:coord["bottom"], image.shape[1]-1] = 255
        new_image[coord["top"]:coord["bottom"], 0] = 255
        new_image[0, coord["left"]:coord["right"]] = 255
        new_image[image.shape[0]-1, coord["left"]:coord["right"]] = 255
    return new_image


def stretch_rectangle(image: np.array, coord: dict) -> np.array:
    rectangle_height = coord["bottom"] - coord["top"]
    rectangle_width = coord["right"] - coord["left"]
    new_image = np.copy(image)
    if rectangle_width > rectangle_height:
        new_image[coord["top"]:coord["bottom"], :] = 255
    elif rectangle_width < rectangle_height:
        new_image[:, coord["left"]:coord["right"]] = 255
    else:
        new_image[coord["top"]:coord["bottom"], :] = 255
        new_image[:, coord["left"]:coord["right"]] = 255
    return new_image