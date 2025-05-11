import streamlit as st

from lib.bounding_box_drawer import IBoundingBoxDrawer, BoundingBoxDrawer
from lib.image_processor import IImageProcessor, ImageProcessor


class NekognitionApp:
    def __init__(
        self,
        bounding_box_drawer: IBoundingBoxDrawer,
        image_processor: IImageProcessor,
        app_title: str = "Nekognition",
        app_sub_title: str = "猫検出アプリケーション",
    ):
        self._bounding_box_drawer = bounding_box_drawer
        self._image_processor = image_processor
        self._app_title = app_title
        self._app_sub_title = app_sub_title

    def run(self):
        st.set_page_config(page_title=self._app_title)
        st.title(self._app_title)
        st.subheader(self._app_sub_title)

        uploaded_file = st.file_uploader(
            "画像をアップロードしてください",
            type=["jpeg", "png"],
            accept_multiple_files=False,
        )

        if uploaded_file is not None:
            self._image_processor.set_uploaded_file(uploaded_file)
            st.markdown("### 検出されたラベル:")  # 検出結果の見出し

            self._image_processor.display_processed_image(
                self._bounding_box_drawer
            )


if __name__ == "__main__":
    app = NekognitionApp(BoundingBoxDrawer(), ImageProcessor())
    app.run()
