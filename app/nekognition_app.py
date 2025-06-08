from PIL import Image
import io

from streamlit.runtime.uploaded_file_manager import UploadedFile
import boto3
from mypy_boto3_rekognition.type_defs import (
    DetectLabelsResponseTypeDef,
    FaceDetailTypeDef,
)
import streamlit as st

from lib.boundary_draw.drawer import BoundingBoxDrawer, IBoundaryDrawer
from lib.rekognition.wrapper import IRekognitionClientWrapper, RekognitionClientWrapper
from lib.face_mosaic_drawer import EllipseFaceMosaicDrawer, IFaceMosaicDrawer
from lib.image_processor import ImageProcessor
from lib.rekognition.utils import extract_cat_label, get_cat_instance_name_and_confidence


class NekognitionApp:
    def __init__(
        self,
        app_title: str = "Nekognition",
        app_sub_title: str = "猫検出アプリケーション",
        rekognition_client: IRekognitionClientWrapper = RekognitionClientWrapper(
            boto3.client("rekognition", "ap-northeast-1")
        ),
        mosaic_drawer: IFaceMosaicDrawer = EllipseFaceMosaicDrawer(),
        bounding_box_drawer: IBoundaryDrawer = BoundingBoxDrawer()
    ):
        self._app_title = app_title
        self._app_sub_title = app_sub_title
        self._rekognition_client = rekognition_client
        self._image_processor = ImageProcessor(
            mosaic_drawer, bounding_box_drawer
        )

    def _update_session_state_with_detection(self, uploaded_file: UploadedFile):
        """画像バイトとRekognitionの検出結果をセッションに保存"""
        st.session_state["image_bytes"] = uploaded_file.read()
        st.session_state["uploaded_filename"] = uploaded_file.name
        st.session_state["detect_face_res"] = self._rekognition_client.detect_faces(
            st.session_state["image_bytes"]
        )
        st.session_state["detect_cats_res"] = self._rekognition_client.detect_cats(
            st.session_state["image_bytes"]
        )

    def run(self):
        """
        Nekognitionのエントリーポイント

        - ページタイトルやサブタイトルなどのヘッダーを表示
        - 画像ファイルのアップロードフォームを表示
        - 画像がアップロードされた場合、Amazon Rekognitionで顔と猫を検出
        - 顔領域にモザイク処理を適用
        - 猫ごとにラベルとハイライト用チェックボックスを表示
        - チェックボックスの状態に応じて枠線の色を切り替えて画像を描画
        - 猫が検出されなかった場合はその旨を表示
        """
        #### ヘッダー類 ####
        st.set_page_config(page_title=self._app_title)
        st.title(self._app_title)
        st.subheader(self._app_sub_title)
        ###################

        #### ファイルアップロードフォーム #####
        uploaded_file = st.file_uploader(
            "画像をアップロードしてください",
            type=["jpeg", "png"],
            accept_multiple_files=False,
        )
        ####################################

        if uploaded_file is not None:
            #### 検出結果見出し ####
            st.markdown("### 検出されたラベル:")
            #######################

            # 枠線ハイライトのチェックボックスが更新された時、Rekognitionへの不要なリクエストを防ぐ
            if "image_bytes" not in st.session_state or st.session_state.get("uploaded_filename") != uploaded_file.name:
                self._update_session_state_with_detection(uploaded_file)

            # sessionから画像バイトとRekognitionからのレスポンスを取り出す
            detect_face_res: list[FaceDetailTypeDef] = st.session_state["detect_face_res"]
            detect_cats_res: DetectLabelsResponseTypeDef = st.session_state["detect_cats_res"]
            image_bytes: bytes = st.session_state["image_bytes"]

            # 検出されたラベル毎に対応する枠線のハイライト有無を管理する Ex.：{"Cat-1": True, "Cat-2": False}
            highlight_states: dict[str, bool] = {}

            cat_label = extract_cat_label(detect_cats_res)
            if cat_label is None or "Instances" not in cat_label:
                #### 検出結果（猫なし） ####
                st.write("アップロードされた画像において猫は検出されませんでした")
                ##########################

            else:
                #### 検出結果（猫あり） 表示形式: Cat-<連番> | 枠線ハイライト用チェックボックス ####
                for index, instance in enumerate(cat_label["Instances"]):
                    instance_name, instance_confidence = get_cat_instance_name_and_confidence(
                        index, instance)
                    col1, col2 = st.columns([1, 1])
                    with col1:
                        st.write(f"- {instance_name} ({instance_confidence})")
                    with col2:
                        highlight_states[instance_name] = st.checkbox(
                            f"highlighten border-line ({instance_name})",
                            value=False,
                            key=instance_name,
                        )
                ############################################################################

            #### 処理済みの画像を表示 ####
            image_file = Image.open(io.BytesIO(image_bytes))
            fig, ax = self._image_processor.process_image(
                image_file,
                detect_face_res,
                detect_cats_res,
                highlight_states,
            )
            st.pyplot(fig)
            ############################
