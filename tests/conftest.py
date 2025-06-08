import pytest
from lib.face_mosaic_drawer import EllipseFaceMosaicDrawer
from lib.boundary_draw.drawer import BoundingBoxDrawer


@pytest.fixture
def mosaic_drawer():
    return EllipseFaceMosaicDrawer()


@pytest.fixture
def box_drawer():
    return BoundingBoxDrawer()
