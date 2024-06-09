def get_bbox_height(bbox):
    return bbox[3] - bbox[1]

def get_bbox_width(bbox):
    return bbox[2] - bbox[0]

def get_bbox_center(bbox):
    return (int((bbox[0] + bbox[2]) / 2), int((bbox[1] + bbox[3]) / 2))

def hardhat_is_on(bbox, bbox_hh, frame):
    # Checks if a hard hat is on a person
    frame_width = frame.shape[1]
    person_height = get_bbox_height(bbox)
    person_width = get_bbox_width(bbox)
    hardhat_height = get_bbox_height(bbox_hh)

    x1 = bbox[0] + (person_width / 2) - (person_width / 6)
    if x1 < 0: x1 = 0
    x2 = bbox[0] + (person_width / 2) + (person_width / 6)
    if x2 > frame_width: x2 = frame_width
    y1 = bbox[1] - hardhat_height / 2
    if y1 < 0: y1 = 0
    y2 = bbox[3] - person_height * 3 / 4
    if y2 < 0: y1 = 0

    person_safe_area = (x1, y1, x2, y2)

    hardhat_position = get_bbox_center(bbox_hh)
    return point_in_area(hardhat_position, person_safe_area)

def point_in_area(point, area):
    # Checks if point is in area
    if area[0] <= point[0] <= area[2] and area[1] <= point[1] <= area[3]:
        return True
    else:
        return False
