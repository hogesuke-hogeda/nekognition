from lib.boundary_draw.utils import calculate_left_top, generate_box


def test_calculate_left_top():
    box = {"Left": 0.1, "Top": 0.2, "Width": 0.3, "Height": 0.4}
    left, top = calculate_left_top(box, 100, 200)
    assert left == 10
    assert top == 40


def test_generate_box():
    box = {"Left": 0.1, "Top": 0.2, "Width": 0.3, "Height": 0.4}
    left, top = 10, 40
    rect = generate_box(box, left, top, 100, 200, "red")
    assert rect.get_x() == 10
    assert rect.get_y() == 40
    assert rect.get_width() == 30
    assert rect.get_height() == 80
    assert rect.get_edgecolor() is not None
