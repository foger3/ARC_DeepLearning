import numpy as np
import matplotlib.pyplot as plt
from scipy import ndimage
import itertools
import torch
import torch.nn as nn

class FullyConnectedNN(nn.Module):
    def __init__(self):
        super(FullyConnectedNN, self).__init__()
        self.flatten = nn.Flatten()  
        self.fc1 = nn.Linear(5 * 10 * 10, 200) 
        self.relu1 = nn.ReLU()  
        self.fc2 = nn.Linear(200, 200) 
        self.relu2 = nn.ReLU() 
        self.fc3 = nn.Linear(200, 200) 
        self.relu3 = nn.ReLU() 
        self.fc4 = nn.Linear(200, 10 * 10) 
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.flatten(x)
        x = self.fc1(x)
        x = self.relu1(x)
        x = self.fc2(x)
        x = self.relu2(x)
        x = self.fc3(x)
        x = self.relu3(x)
        x = self.fc4(x)
        x = self.sigmoid(x)
        return x.view(-1, 10, 10)
    

def get_test_metrics(x_test, y_test, model, criterion, method_index_test, analogy_breakdown):
    model.eval()
    with torch.no_grad():
        y_pred = model(x_test)
        test_loss = criterion(y_pred, y_test).item()
        y_pred = torch.round(y_pred)
        elements_equal = np.equal(y_test, y_pred)
        rows_equal_all = [torch.all(rows_equal.flatten()) for rows_equal in elements_equal]
        percentage_solved_total = (np.sum(rows_equal_all) / y_test.shape[0]) * 100
        if not analogy_breakdown:
            return {"test loss": test_loss, "percent_solved": percentage_solved_total}
        else:
            analogy_metrics = {}
            method_names = np.unique(method_index_test)
            for method in method_names:
                y_pred = model(x_test[method_index_test == method])
                y_pred = torch.round(y_pred)
                elements_equal = np.equal(y_test[method_index_test == method], y_pred)
                rows_equal_all = [torch.all(rows_equal.flatten()) for rows_equal in elements_equal]
                percentage_same = (np.sum(rows_equal_all) / y_test[method_index_test == method].shape[0]) * 100
                analogy_metrics[method] = percentage_same
            return {"test loss": test_loss, "percent_solved": percentage_solved_total} | analogy_metrics



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


def paint_edge(image: np.array, furthest = True) -> np.array:
    height, width = image.shape
    white_pixels = np.where(image == 255)
    coord = {"bottom": max(white_pixels[0]), "top": min(white_pixels[0]), "left": min(white_pixels[1]), "right": max(white_pixels[1])}
    top_distance = coord["top"]
    bottom_distance = height - coord["bottom"] - 1
    left_distance = coord["left"]
    right_distance = width - coord["right"] - 1
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


def create_image(img_size: int, shape: str) -> np.array:
    new_image = np.zeros((img_size, img_size), dtype=np.uint8)
    if shape == "L":
        min_size = 2
    else:
        min_size = 1
    top_left = {"top": np.random.randint(1, img_size-min_size), "left": np.random.randint(1, img_size-min_size)}
    bottom_right = {"bottom": np.random.randint(top_left["top"] + min_size, img_size), "right": np.random.randint(top_left["left"] + min_size, img_size)}
    coord = top_left | bottom_right
    new_image[coord["top"]:coord["bottom"], coord["left"]:coord["right"]] = 255
    if shape == "rectangle":
        pass
    elif shape == "L":
        rec_width = coord["right"] - coord["left"]
        rec_height = coord["bottom"] - coord["top"]
        keep_left = np.random.choice([True, False])
        keep_bottom = np.random.choice([True, False])
        side_trimming = np.random.choice(range(1, rec_width))
        vertical_trimming = np.random.choice(range(1, rec_height))
        if keep_left and keep_bottom:
            new_image[coord["top"]:(coord["bottom"] - vertical_trimming), (coord["left"]+side_trimming):coord["right"]] = 0
        elif keep_left and (not keep_bottom):
            new_image[(coord["top"]+ vertical_trimming):coord["bottom"], (coord["left"]+side_trimming):coord["right"]] = 0
        elif (not keep_left) and keep_bottom:
            new_image[coord["top"]:(coord["bottom"] - vertical_trimming), coord["left"]:(coord["right"] - side_trimming)] = 0
        else:
            new_image[(coord["top"]+ vertical_trimming):coord["bottom"], coord["left"]:(coord["right"] - side_trimming)] = 0
    else:
        raise ValueError(f"Illegal shape: {shape}")
    return new_image


def gravity(image: np.array,  direction: str) -> np.array:
    vert_shift = 0
    hori_shift = 0
    new_image = np.zeros((image.shape[0], image.shape[0]), dtype=np.uint8)
    white_pixels = np.where(image == 255)
    coord = {"bottom": max(white_pixels[0]), "top": min(white_pixels[0]), "left": min(white_pixels[1]), "right": max(white_pixels[1])}
    if direction == "down":
        vert_shift = image.shape[0]-coord["bottom"]-1
    elif direction == "up":
        vert_shift = -1 * coord["top"]
    elif direction == "left":
        hori_shift = -1 * coord["left"]
    elif direction == "right":
        hori_shift = image.shape[1]-coord["right"]-1
    else:
        raise ValueError("gravity direction has to be one of left, right, up, down")
    for row in range(image.shape[0]):
        for col in range(image.shape[1]):
            if image[row, col] == 255:
                if (row + vert_shift < image.shape[0]) and (col + hori_shift < image.shape[1]):
                    new_image[row+vert_shift, col+hori_shift] = 255
    return new_image


def invalid_matrix(matrix: np.array, height: int, width:int, number_rectangles: int) -> np.array:
    if matrix.shape != (height, width):
        return True #if wrong shape
    top_left_value = matrix[0, 0]
    if not np.all(matrix[0, :] == top_left_value) or not np.all(matrix[-1, :] == top_left_value) or not np.all(matrix[:, 0] == top_left_value) or not np.all(matrix[:, -1] == top_left_value):
        return True #if figure(s) touches edges
    _, num_features = ndimage.label(matrix)
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
    plt.show()


def mirror_image(image: np.array, horizontal=True) -> np.array:
    if horizontal:
        mirrored_image = np.fliplr(image)
    else:
        mirrored_image = np.flipud(image)
    return mirrored_image


def move(image: np.array, horizontal_movement: int, vertical_movement: int) -> np.array:
    height, width = image.shape
    new_image = np.zeros((height, width), dtype=np.uint8)
    non_zero_coords = np.argwhere(image > 0)
    new_non_zero_coords = [[row + horizontal_movement, col + vertical_movement] for row, col in non_zero_coords if (0 <= row+horizontal_movement <  image.shape[0]) and (0 <= col + vertical_movement < image.shape[1])]
    for r, c in new_non_zero_coords:
        new_image[r, c] = 255
    return new_image


def grow(image: np.array, top_add: int, bottom_add: int, left_add: int, right_add: int) -> np.array:
    new_image = np.copy(image)
    for row in range(image.shape[0]):
        for col in range(image.shape[1]):
            if image[row,col] == 255:
                new_image[row,col] = 255
            elif any(image[row, max(0, col-right_add):col] == 255):
                new_image[row, col] = 255
            elif any(image[row, col:min((col+left_add + 1), image.shape[1])] == 255):
                new_image[row, col] = 255
            elif any(image[max(0,(row-bottom_add)):row, col] == 255):
                new_image[row, col] = 255
            elif any(image[row:min((row+top_add + 1), new_image.shape[0]), col] == 255):
                new_image[row, col] = 255
            else:
                pass
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

#L approved
def draw_shadows(image: np.array, reverse: bool) -> np.array:
    new_image = np.zeros((image.shape[0], image.shape[1]), dtype=np.uint8)
    white_pixels = np.where(image == 255)
    coord = {"bottom": max(white_pixels[0]), "top": min(white_pixels[0]), "left": min(white_pixels[1]), "right": max(white_pixels[1])}
    if reverse:
        new_image[0:coord["top"], image.shape[1]-1] = 255
        new_image[coord["bottom"]+1:image.shape[0], image.shape[1]-1] = 255
        new_image[0:coord["top"], 0] = 255
        new_image[coord["bottom"]+1:image.shape[0], 0] = 255
        new_image[0, 0:coord["left"]] = 255
        new_image[0, coord["right"]+1:image.shape[1]] = 255
        new_image[image.shape[0]-1, 0:coord["left"]] = 255
        new_image[image.shape[0]-1, coord["right"]+1:image.shape[1]] = 255
    else:
        new_image[coord["top"]:coord["bottom"]+1, image.shape[1]-1] = 255
        new_image[coord["top"]:coord["bottom"]+1, 0] = 255
        new_image[0, coord["left"]:coord["right"]+1] = 255
        new_image[image.shape[0]-1, coord["left"]:coord["right"]+1] = 255
    return new_image

#L approved
def stretch_rectangle(image: np.array) -> np.array:
    white_pixels = np.where(image == 255)
    coord = {"bottom": max(white_pixels[0]), "top": min(white_pixels[0]), "left": min(white_pixels[1]), "right": max(white_pixels[1])}
    rectangle_height = coord["bottom"] - coord["top"]
    rectangle_width = coord["right"] - coord["left"]
    new_image = np.copy(image)
    if rectangle_width > rectangle_height:
        new_image[coord["top"]:coord["bottom"] + 1, :] = 255
    elif rectangle_width < rectangle_height:
        new_image[:, coord["left"]:coord["right"] + 1] = 255
    else:
        new_image[coord["top"]:coord["bottom"] + 1, :] = 255
        new_image[:, coord["left"]:coord["right"] + 1] = 255
    return new_image


def count_pixels(image: np.array, left_right: bool, top_bottom: bool) -> np.array:
    new_image = np.zeros((image.shape[0], image.shape[0]), dtype=np.uint8)
    for i in range(sum(sum(image == 255))):
        col = i % image.shape[1] if left_right else int(i / image.shape[1])
        row = int(i / image.shape[0]) if left_right else i % image.shape[0]
        if top_bottom:
            new_image[row, col] = 255
        else:
            new_image[image.shape[0] - row - 1, col] = 255
    return new_image